import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.ndimage import zoom
from scipy.special import comb  # n choose k
import xarray as xr
import cv2
from tqdm.notebook import tqdm


class para_mcam:
    def __init__(self, stack, recon_shape, dxyz, batch_across_images=True, batch_size=None, xyz_offset=np.zeros(3),
                 scale=1, depth_of_field=None, interp_rays=True, dither_rays=True, occupancy_grid=None,
                 ):
        # stack: 3D tensor of shape num_images, num_rows, num_cols, num_channels.
        # recon_shape: the pixel dimensions of the 3D reconstruction.
        # dxyz: pixel size of recon.
        # batch_size is the stratified batch size, meaning every angle will have the same exact spatial positions.
        # batch_across_images: if true, each batch is a random selection of images; if false, random pixels.
        # scale: recon_shape * scale is the actual shape.
        # depth_of_field: the confocal parameter of a Gaussian beam; multiply by 1/(1+(z/(dof/2))^2) before summing for
        # the projection. In um.
        # interp_rays: when sampling the reconstruction, whether to interpolate across at 2x2x2 neighborhood.
        # dither_rays: whether to sample uniformly along the rays or add a random uniform value between 0 and 1.
        # occupancy_grid: a boolean 4D tensor with the same spatiotemporal extent as the full reconstruction, but
        # downsampled (e.g., 1201x64x64x64 -- time first, since we'll be indexing by time primarily).

        self.stack = stack
        self.num_channels = self.stack.shape[3]  # number of channels; stack must at least have a singleton dim 3
        self.num_images = self.stack.shape[0]  # number of images in dataset
        self.recon_shape_base = recon_shape
        self.dxyz = dxyz  # recon pixel size in um
        self.step = dxyz  # step size between points along ray
        self.xyz_offset = xyz_offset
        self.batch_size = batch_size
        self.batch_across_images = batch_across_images
        self.scale = scale
        self.sig_proj = .42465  # for the intepolation kernel width
        self.optimizer = tf.keras.optimizers.Adam
        self.prefetch = 1  # how many batches to prefetch?
        self.num_z_steps = int(np.prod(np.array(recon_shape) * scale) ** (1/3))  # number of points along
        # backprojection line, computed as geometric mean of reconstruction dimensions
        self.force_positive_recon = True
        self.extra_finetune_order = None
        self.depth_of_field = depth_of_field
        self.interp_rays = interp_rays
        self.dither_rays = dither_rays
        if type(occupancy_grid) == np.ndarray:
            self.occupancy_grid = tf.constant(occupancy_grid)
        else:
            self.occupancy_grid = occupancy_grid

        self.ckpt = None  # for checkpointing models
        self.lr_dict = None  # for zeroing and reseting learning rates

    def generate_dataset(self, stack_downsamp=None, prefetch=-1, identifier=None, preshuffle=False, shuffle_buffer=None,
                         seed=0):
        # batching is always done, and is done in a stratified manner.
        # if stack_downsamp is None, then use self.stack_downsamp; else, use stack_downsamp.
        # identifier: if supplied, then let the dataset yield this value at every iteration; this is useful if you're
        # using combining multiple tf.datasets and need to know which dataset you're sampling from.
        # preshuffle: shuffle before passing to tf.data.
        # shuffle_buffer: if None, then defaults to full dataset.
        # seed is for preshuffle.

        if preshuffle or shuffle_buffer is not None:
            assert not self.batch_across_images  # didn't bother implementing for this case

        if stack_downsamp is None:
            stack_downsamp = self.stack_downsamp

        if identifier is not None:  # will indicate to gradient_update whether to unpack an identifier
            self.yield_identifier = True
        else:
            self.yield_identifier = False

        if self.batch_size is None:
            # no need to generate batch; just use self.stack_downsamp, self.galvo_xy_downsamp
            print('Dataset not needed')
            return None
        else:
            if self.batch_across_images:
                # sample a subset of the images, and keep track of the indices downsampled so that you can gather the
                # corresponding variables
                tensor_slices = [stack_downsamp, np.arange(self.num_images, dtype=np.int32)]
                dataset = (tf.data.Dataset.from_tensor_slices(tuple(tensor_slices)).shuffle(self.num_images)
                           .batch(self.batch_size, drop_remainder=True).repeat(None).prefetch(prefetch))
                return dataset
            else:
                num_pixels = np.prod(self.im_downsampled_shape)

                if shuffle_buffer is None:
                    shuffle_buffer = num_pixels

                # transpose to batch along space, not image number;
                stack_downsamp_T = np.transpose(stack_downsamp, (1, 0, 2))
                # need to also get coordinates of the spatial positions to index into pixel-wise background:
                tensor_slices = [stack_downsamp_T, self.galvo_xy_downsamp,
                                 np.arange(num_pixels, dtype=np.int32)]
                if identifier is not None:
                    tensor_slices.append(np.tile(identifier, num_pixels))
                    # this is a little wasteful, since you're yielding as many identifiers as there are pixels in a
                    # batch, but you only need 1 ...
                if preshuffle:
                    np.random.seed(seed)
                    preshuffle_inds = np.random.permutation(num_pixels)
                    for i in range(len(tensor_slices)):
                        tensor_slices[i] = tensor_slices[i][preshuffle_inds]
                dataset = (tf.data.Dataset.from_tensor_slices(tuple(tensor_slices))
                           .shuffle(shuffle_buffer).batch(self.batch_size, drop_remainder=True)
                           .repeat(None).prefetch(prefetch))  # I believe this code requires fixed batch dim size
                return dataset

    def create_variables(self, nominal_probe_xy, propagation_model='parabolic', inds_keep=None, learning_rates=None,
                         variable_initial_values=None, stack_downsample_factor=None, recon=None,
                         use_attenuation_map=False
                         ):
        # nominal_probe_xy: _ by 2 array of the programmed xy coordinate trajectory of the probe.
        # note that "galvo" refers to the equivalent point-scanning system, with a galvo replacing the aperture.
        # similarly, "probe" refers to the camera sensor positions, analogous to the scanning probe positions for ocrt.
        # stack_downsample_factor is an integer that downsamples the stack and coordinates. If None, then it will be
        # computed from self.scale.
        # recon: optional initial estimate.
        # inds_keep: if not all cameras are to be used. This will affect the shape of the tf.Variables. Alternatively,
        # can just apply inds_keep outside.
        # use_attenuation_map: optimize an attenuation coefficient map of the same size as the reconstruction.

        if inds_keep is not None:
            self.stack = self.stack[inds_keep]
            self.num_images = len(inds_keep)
            nominal_probe_xy = nominal_probe_xy[inds_keep]
        self.inds_keep = inds_keep

        # define downsample factor:
        self.recon_shape = np.int32(np.array(self.recon_shape_base) * self.scale)
        self.recon_shape_with_channels = (self.recon_shape[0], self.recon_shape[1], self.recon_shape[2],
                                          self.num_channels)
        if stack_downsample_factor is None:
            self.downsample = np.int32(1 / self.scale)  # also downsample the images to save computation;
            self.downsample = np.maximum(self.downsample, 1)  # obviously can't downsample with 0;
        else:
            self.downsample = stack_downsample_factor
        self.im_downsampled_shape = np.array([(self.stack.shape[1] - 1) // self.downsample + 1,
                                              (self.stack.shape[2] - 1) // self.downsample + 1])

        self.propagation_model = propagation_model
        self.nominal_probe_xy = nominal_probe_xy
        self.dxyz /= self.scale
        self.recon_fov = self.dxyz * self.recon_shape

        self.use_attenuation_map = use_attenuation_map

        # dictionaries of tf.Variables and their corresponding optimizers:
        self.train_var_dict = dict()
        self.optimizer_dict = dict()
        self.non_train_dict = dict()  # dict of variables that aren't trained (probably .assigned()'d; for checkpoints)
        self.tensors_to_track = dict()  # intermediate tensors to track; have a tf.function return the contents

        def use_default_for_missing(input_dict, default_dict):
            # to be used directly below; allows for dictionaries in which not all keys are specified; if not specified,
            # then use default_dict's value.

            if input_dict is None:  # if nothing given, then use the default
                return default_dict
            else:
                for key in default_dict:
                    if key in input_dict:
                        if input_dict[key] is None:  # if the key is present, but None is specified
                            input_dict[key] = default_dict[key]
                        else:  # i.e., use the value given
                            pass
                    else:  # if key is not even present
                        input_dict[key] = default_dict[key]
                return input_dict

        # define variables to be optimized, based on a dictionary of variable names and learning rates/initial values;
        # these variables are always used; otherwise, add additional variables via the propagation_model:
        # (negative learning rates mean that later we will not update these variables)
        default_learning_rates = {'f_mirror': -1e-3, 'f_lens': 1e-3,
                                  'galvo_xy': 1e-3, 'galvo_normal': 1e-3, 'galvo_theta': 1e-3,
                                  'probe_dx': 1e-3, 'probe_dy': 1e-3, 'probe_z': 1e-3, 'probe_normal': 1e-3,
                                  'probe_theta': 1e-3, 'per_image_scale': 1e-3, 'per_image_bias': 1e-3,
                                  }
        default_variable_initial_values = {'f_mirror': 25.4,  # focal length  of parabolic mirror in mm
                                           'f_lens': 25,  # effective focal length of lens before the mirror in mm
                                           'galvo_xy': 4.5 / 2,  # isotropic scan amplitude at lens in mm
                                           # principal plane in x in mm
                                           'galvo_normal': np.array((1e-7, 1e-7, -1), dtype=np.float32),  # direction
                                           # of the center ray
                                           'galvo_theta': 0,  # angle-vector representation of rotation
                                           'probe_dx': 0,  # global x shift in the nominal probe trajectories
                                           'probe_dy': 0,  # global y shift in the nominal probe trajectories
                                           'probe_z': 25,  # distance between the lens and mirror foci (the origin)
                                           'probe_normal': np.array((1e-7, -1), dtype=np.float32),  # normal of
                                           # the probe translation plane, in case there's any relative tilt
                                           'probe_theta': 0,  # angle-vector representation of probe translation plane
                                           'per_image_scale': np.ones(self.num_images, dtype=np.float32),
                                           'per_image_bias': np.zeros(self.num_images, dtype=np.float32),
                                           }

        # WARNING: if you define a variable in these dicts, make sure they are used, or the optimizers might do nothing!

        # if there are additional variables, define here (modify the two dictionaries):
        if 'nonparametric' in propagation_model:
            # allow the final boundary conditions to vary arbitrarily:
            default_learning_rates = {**default_learning_rates, 'delta_r': 1e-3, 'delta_u': 1e-3, 'r_2nd_order': 1e-3,
                                      'u_2nd_order': 1e-3}
            default_variable_initial_values = {**default_variable_initial_values,
                                               'delta_r': np.zeros((self.num_images, 3), dtype=np.float32),  # position
                                               'delta_u': np.zeros((self.num_images, 3), dtype=np.float32),  # orient
                                               'r_2nd_order': np.zeros((self.num_images, 5, 3)),  # 2nd order
                                               # correction of A-scan position; 3 sets of 5 coefficients:
                                               # a0*x + a1*y + a2*x**2 + a3*y**2 + a4*x*y; (one for x,y,z); allowing
                                               # this for xy allows for nonlinearly spaced A-scans
                                               'u_2nd_order': np.zeros((self.num_images, 5, 3)),  # likewise, for
                                               # angular fanning
                                               }

        if 'nonparametric_higher_order_correction' in propagation_model:
            # try 3rd and 4th order correction on top of the 2nd order
            assert 'nonparametric' in propagation_model
            default_learning_rates = {**default_learning_rates, 'r_higher_order': 1e-3, 'u_higher_order': 1e-3}
            default_variable_initial_values = {**default_variable_initial_values,
                                               'r_higher_order': np.zeros((self.num_images, 9, 3)),
                                               # 3rd and 4th order correction of A-scan position; 3 sets of 9 coeffs:
                                               # a0*x**3 + a1*y**3 + a2*x**2*y + a3*x*y**2 + ...
                                               # a4*x**4 + a5*y**4 + a6*x**3*y + a7*x**2*y**2 + a8*x*y**3
                                               'u_higher_order': np.zeros((self.num_images, 9, 3)),  # likewise, for
                                               # angular fanning
                                               }


        if 'extra_finetune' in propagation_model:
            # polynomial model, but applied AFTER refracting thru nmr tube, and user can specify arbitrary number of
            # terms
            assert self.extra_finetune_order is not None  # should be an integer
            order = self.extra_finetune_order

            self.num_extra_terms = int(comb(order + 2, 2))

            self.extra_orders = list()
            for order_ in range(order + 1):
                for exp in range(order_ + 1):
                    self.extra_orders.append((exp, (order_ - exp)))

            assert len(self.extra_orders) == self.num_extra_terms

            self.extra_orders = np.array(self.extra_orders)  # _ by 2

            default_learning_rates = {**default_learning_rates, 'r_extra': 1e-3, 'u_extra': 1e-3}
            default_variable_initial_values = {**default_variable_initial_values,
                                               'r_extra': np.zeros((self.num_images, self.num_extra_terms, 3)),
                                               'u_extra': np.zeros((self.num_images, self.num_extra_terms, 3)),
                                               }

        if 'nmr_tube' in propagation_model:
            default_learning_rates = {**default_learning_rates, 'nmr_outer_radius': -1e-3, 'nmr_inner_radius': -1e-3,
                                      'nmr_delta_r': 1e-3, 'nmr_normal': 1e-3, 'nmr_theta': 1e-3,
                                      'n_glass': -1e-3, 'n_medium': -1e-3}
            default_variable_initial_values = {**default_variable_initial_values,
                                               'nmr_outer_radius': np.float32(4.9635 / 2),  # in mm; for thicker version
                                               'nmr_inner_radius': np.float32(3.4300 / 2),  # in mm
                                               'nmr_delta_r': np.zeros(3, dtype=np.float32),  # centered at para focus
                                               'nmr_normal': np.array((1e-7, 1e-7, -1), dtype=np.float32),
                                               'nmr_theta': np.float32(0),
                                               'n_glass': np.float32(1.5),
                                               'n_medium': np.float32(1.33)
                                               }

        default_learning_rates = {**default_learning_rates, 'recon': 1e-3}
        if recon is None:
            recon_init = np.zeros(self.recon_shape_with_channels)
        else:  # otherwise, upsample to the current shape (trilinear interpolation)
            recon_init = zoom(recon, self.recon_shape_with_channels / np.array(recon.shape), order=2)
        default_variable_initial_values = {**default_variable_initial_values, 'recon': recon_init}

        if use_attenuation_map:
            default_learning_rates = {**default_learning_rates, 'attenuation_map': 1e-2}
            default_variable_initial_values = {**default_variable_initial_values,
                                               'attenuation_map': np.zeros(self.recon_shape,
                                                                           dtype=np.float32)}

        learning_rates = use_default_for_missing(learning_rates, default_learning_rates)
        variable_initial_values = use_default_for_missing(variable_initial_values, default_variable_initial_values)
        if 'extra_finetune' in propagation_model:  # this gives the option to use more terms after earlier optimization
            r_extra = variable_initial_values['r_extra'].copy()
            if r_extra.shape[1] != self.num_extra_terms:
                expanded = np.zeros((self.num_images, self.num_extra_terms, 3))
                expanded[:, :r_extra.shape[1], :] = r_extra
                variable_initial_values['r_extra'] = expanded
                print('padded r_extra')
            u_extra = variable_initial_values['u_extra'].copy()
            if u_extra.shape[1] != self.num_extra_terms:
                expanded = np.zeros((self.num_images, self.num_extra_terms, 3))
                expanded[:, :u_extra.shape[1], :] = u_extra
                variable_initial_values['u_extra'] = expanded
                print('padded u_extra')

        self.learning_rates = learning_rates  # save for monitoring purposes
        self.variable_initial_values = variable_initial_values

        if 'delta_r' in variable_initial_values:
            # for keeping the mean shift 0, if optimizing this
            self.delta_r_initial = np.mean(variable_initial_values['delta_r'], axis=0)[None]

        # create variables and train ops based on the initial values:
        for key in learning_rates.keys():  # learning_rates and variable_initial_values should have the same keys
            var = tf.Variable(variable_initial_values[key], dtype=tf.float32, name=key)
            opt = self.optimizer(learning_rate=learning_rates[key])
            self.train_var_dict[key] = var
            self.optimizer_dict[key] = opt

        # coordinates of the IDEAL full boundary conditions (i.e., for all A-scans); to be indexed with batch_inds
        # they will be transformed according to the tf.Variables describing misalignment
        self.probe_xy_BC = self.nominal_probe_xy.astype(np.float32)

        # coordinates of the images:
        x = np.linspace(-1, 1, self.stack.shape[2])
        y = np.linspace(1, -1, self.stack.shape[1])
        y *= self.stack.shape[1] / self.stack.shape[2]  # isotropic pixels
        x, y = np.meshgrid(x, y, indexing='ij')
        self.galvo_xy_base = np.stack([x, y]).T.astype(np.float32)
        # base shape before downsampling^

        # downsample galvo_xy coordinates and stack:
        self.galvo_xy_downsamp = self.galvo_xy_base[::self.downsample, ::self.downsample, :].reshape(-1, 2)
        self.stack_downsamp = self.format_stack(self.stack)
        # ^spatial dims are flattened

        # Create a list of booleans to accompany self.train_var_list and self.optimizer_list to specify whether to train
        # those variables (as specified by whether the user-specified learning rates are negative). Doing this so
        # that autograph doesn't traverse all branches of the conditionals. If the user ever wants to turn off
        # optimization of a variable mid-optimization, then just do .assign(0) to the learning rate, such that the
        # update is still happening, but the change is 0.
        self.trainable_or_not = list()
        for var in self.train_var_dict:
            name = self.train_var_dict[var].name[:-2]
            flag = learning_rates[name] > 0
            self.trainable_or_not.append(flag)

        if self.depth_of_field is not None:
            line = tf.range(self.num_z_steps, dtype=tf.float32) * self.step / self.scale
            line -= tf.reduce_mean(line)
            self.z_int_profile = 1 / (1 + (line / (self.depth_of_field / 2)) ** 2)  # multiply by this before projecting
            if self.occupancy_grid is not None:
                z_int_profile = self.z_int_profile.numpy()
                z_int_profile[0] = 0  # can't do assignment in tf
                self.z_int_profile = z_int_profile
        else:
            self.z_int_profile = None
            if self.occupancy_grid is not None:
                raise Exception('for now, depth_of_field and z_int_profile should be defined, because use it to zero '
                                'out the first ray entries')


    def format_stack(self, stack):
        # to be used by create_variables and the user when they want to use different stacks
        if self.inds_keep is not None and len(stack) != len(self.inds_keep):
            # detect whether inds_keep has been applied;
            stack = stack[self.inds_keep]
        stack_reformat = stack[:, ::self.downsample,
                                  ::self.downsample, :].reshape(self.num_images, -1, self.num_channels)

        return stack_reformat

    def _axis_angle_rotmat(self, axis, angle):
        # return 3D rotation matrix given axis and angle.
        # axis is of shape (3) and angle is a single number.

        axis_unit, _ = tf.linalg.normalize(axis)  # convert to unit vector
        cos = tf.cos(angle)
        sin = tf.sin(angle)
        ux = axis_unit[0]
        uy = axis_unit[1]
        uz = axis_unit[2]

        r00 = cos + ux ** 2 * (1 - cos)
        r01 = ux * uy * (1 - cos) - uz * sin
        r02 = ux * uz * (1 - cos) + uy * sin
        r10 = ux * uy * (1 - cos) + uz * sin
        r11 = cos + uy ** 2 * (1 - cos)
        r12 = uy * uz * (1 - cos) - ux * sin
        r20 = ux * uz * (1 - cos) - uy * sin
        r21 = uy * uz * (1 - cos) + ux * sin
        r22 = cos + uz ** 2 * (1 - cos)

        rotmat = tf.stack([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
        return rotmat

    def _generate_2D_rotmat(self, angle):
        # simple, 2D rotation
        # angle is a 1D vector

        cos = tf.cos(angle)
        sin = tf.sin(angle)

        rotmat = tf.stack([[cos, -sin],
                           [sin, cos]])
        return rotmat  # shape: 2, 2, _;

    def _tf_gather_if_necessary(self, params, batch_inds):
        # this makes the propagation code more compact, so that you don't need this set of embedded if statements every
        # time.
        # if batch_inds is None, then just return params.
        # only if not batching across images is this function anything but the identity function.

        if self.batch_size is None:
            return params
        else:
            if self.batch_across_images:
                return tf.gather(params, batch_inds, axis=0)
            else:
                return params

    def _propagate_to_parabolic_focus(self, batch):
        # generate boundary conditions from parabolic mirror parameters; need position and direction.
        # then, propagate to just before the parabolic mirror's focus, and return the final position and direction,
        # which will serve as the boundary conditions for ray propagation through the sample.
        # batch: can be either batch_inds in the case of batching across images; otherwise, it's the galvo_xy batch.

        # unpack:
        f_mirror = self.train_var_dict['f_mirror']
        f_lens = self.train_var_dict['f_lens']
        galvo_xy = self.train_var_dict['galvo_xy']
        galvo_theta = self.train_var_dict['galvo_theta']
        galvo_normal = self.train_var_dict['galvo_normal']
        probe_dx = self.train_var_dict['probe_dx']
        probe_dy = self.train_var_dict['probe_dy']
        probe_z = self.train_var_dict['probe_z']
        probe_normal = self.train_var_dict['probe_normal']
        probe_theta = self.train_var_dict['probe_theta']
        probe_xy = self.probe_xy_BC

        # here, I'm forcing probe_normal's z coordinate to be 0:
        probe_normal = tf.concat([probe_normal, [0.0]], axis=0)

        # gather the relevant rays to propagate, corresponding to the batch:
        if self.batch_size is None:
            # batch should be None; or even if not, it's ignored
            num_images = self.num_images
            num_pixels = np.prod(self.im_downsampled_shape)
            batch_inds = None
            galvo_xy_batch_ = self.galvo_xy_downsamp
        else:
            if self.batch_across_images:
                num_images = self.batch_size
                num_pixels = np.prod(self.im_downsampled_shape)
                batch_inds = batch  # interpret input to this function as batch inds
                galvo_xy_batch_ = self.galvo_xy_downsamp
            else:
                num_images = self.num_images
                num_pixels = self.batch_size
                batch_inds = None
                galvo_xy_batch_ = batch   # shape: num_pixels, 2
                # ^interpret input to this function as galvo_xy_batch (a random subset of pixels coordinates)

        # deal with probe/camera-invariant image ampltiudes:
        galvo_xy_batch = galvo_xy * galvo_xy_batch_  # multiply by scan amplitude; shape: spatial pixels, 2
        galvo_xy_batch = tf.broadcast_to(galvo_xy_batch, (num_images, num_pixels, 2))
        # shape^: batch_size, spatial, 2

        # add the uniform z coordinate:
        galvo_xyz_batch = tf.concat([galvo_xy_batch, tf.broadcast_to(
            -f_lens, (num_images, num_pixels, 1))], axis=-1)
        # resulting list of vectors should be normalized; batch_size, spatial, 3

        # expand with z coordinate:
        probe_xy_batch = self._tf_gather_if_necessary(probe_xy, batch_inds)  # shape: num_images, 2
        probe_xyz_batch = tf.concat([probe_xy_batch, tf.broadcast_to(probe_z, (num_images, 1))], axis=1)
        # augment with probe_z, because rotation will be done wrt the absolute origin; batch_size by 3

        # initial ray direction:

        # first, rotate according to galvo_normal/galvo_theta.
        # have to rotate rays; fortunately, they're all origin-centered at this stage, so it's the same as rotating
        # points;
        galvo_rotmat = self._axis_angle_rotmat(galvo_normal, galvo_theta)
        galvo_xyz_batch @= galvo_rotmat  # rotate vectors (vectors start at origin,so just need to rotate tips)
        # shape: batch_size, spatial, 3

        # next, account for probe translational plane tilt:
        # to do this, we rotate both ends of each ray:
        ray_start = tf.broadcast_to(probe_xyz_batch[:, None, :],
                                    (num_images, num_pixels, 3))
        # shape^: batch_size, spatial, 3
        ray_end = ray_start + galvo_xyz_batch   # shape: batch_size, spatial, 3
        probe_rotmat = self._axis_angle_rotmat(probe_normal, probe_theta)
        ray_start @= probe_rotmat
        ray_end @= probe_rotmat

        # flatten out batch and spatial dims:
        ray_start = tf.reshape(ray_start, (-1, 3))
        ray_end = tf.reshape(ray_end, (-1, 3))
        self.flattened_length = num_images * num_pixels  # length of first dim of above 2^
        self.ray_start = ray_start
        self.ray_end = ray_end

        # final boundary conditions prior to mirror:
        ray_directions = ray_end - ray_start  # shape: flat length, 3
        ray_directions, _ = tf.linalg.normalize(ray_directions, axis=1)  # normalize to unit vectors
        ray_positions = ray_start  # shape: flat length, 3
        ray_positions = ray_positions + tf.stack([probe_dx, probe_dy, 0])[None, :]

        x0 = ray_positions[:, 0]
        y0 = ray_positions[:, 1]
        z0 = ray_positions[:, 2]
        r0 = ray_positions  # pseudonym

        ux = ray_directions[:, 0]
        uy = ray_directions[:, 1]
        uz = ray_directions[:, 2]
        u0 = ray_directions  # pseudonym

        # propagate to mirror:
        # coefficients of quadratic equation a*d^2 + b*d + c, where d is the distance from r0 to the mirror
        a = -ux ** 2 - uy ** 2
        b = 4 * f_mirror * uz - 2 * ux * x0 - 2 * uy * y0
        c = 4 * f_mirror * z0 - x0 ** 2 - y0 ** 2 + 4 * f_mirror ** 2
        d = 2 * c / (-b + tf.sqrt(b ** 2 - 4 * a * c))

        r_at_mirror = r0 + d[:, None] * u0  # position at mirror

        # propagate to near focus:
        n = tf.concat([-r_at_mirror[:, :2] / 2 / f_mirror,
                       np.ones((self.flattened_length, 1))], 1)  # surface normals
        n, _ = tf.linalg.normalize(n, axis=1)
        u1 = u0 - 2 * tf.reduce_sum(u0 * n, 1, keepdims=True) * n  # the sum is a dot product with broadcasting;
        d_remain = probe_z + 2 * f_mirror - d  # remaining distance to propagate

        r_near_focus = r_at_mirror + u1 * d_remain[:, None]  # position near focus

        # the next boundary conditions for ray prop through sample:
        r_before_sample = r_near_focus * 1000  # convert to um
        u_before_sample = u1
        # shape: num_images * batch_size, 3

        if 'nonparametric' in self.propagation_model:
            r = tf.reshape(r_before_sample, [num_images, num_pixels, 3])  # unflatten
            u = tf.reshape(u_before_sample, [num_images, num_pixels, 3])  # unflatten

            # subtract out mean from delta_r:
            delta_r = (self.train_var_dict['delta_r'] + self.delta_r_initial  # make sure the mean stays the same
                       - tf.reduce_mean(self.train_var_dict['delta_r'], axis=0, keepdims=True))  # shape:  batch_size, 3

            r_new = r + self._tf_gather_if_necessary(delta_r, batch_inds)[:, None, :]
            u_new = u + self._tf_gather_if_necessary(self.train_var_dict['delta_u'], batch_inds)[:, None, :]
            u_new, _ = tf.linalg.normalize(u_new, axis=-1)  # renormalize to unit vector

            # 2nd order correction:
            # adjust position (r):
            xy = galvo_xy_batch_  # spatial, 2
            coefs = self.train_var_dict['r_2nd_order']  # shape: num_images, 5, 3
            coefs = self._tf_gather_if_necessary(coefs, batch_inds)[:, None, :, :]
            x = xy[None, :, 0:1]  # shape: 1, spatial, 1;
            y = xy[None, :, 1:]
            dr = (x * coefs[:, :, 0, :] + y * coefs[:, :, 1, :] + x ** 2 * coefs[:, :, 2, :]
                  + y ** 2 * coefs[:, :, 3, :] + x * y * coefs[:, :, 4, :])
            # shape^: batch_size, spatial, 3
            r_new = r_new + dr
            # adjust ray fans (u):
            coefs = self.train_var_dict['u_2nd_order']  # shape: num_images, 5, 3
            coefs = self._tf_gather_if_necessary(coefs, batch_inds)[:, None, :, :]
            du = (x * coefs[:, :, 0, :] + y * coefs[:, :, 1, :] + x ** 2 * coefs[:, :, 2, :]
                  + y ** 2 * coefs[:, :, 3, :] + x * y * coefs[:, :, 4, :])
            # shape^: batch_size, spatial, 3
            u_new = u_new + du

            if 'higher_order_correction' in self.propagation_model:
                coefs = self.train_var_dict['r_higher_order']
                coefs = self._tf_gather_if_necessary(coefs, batch_inds)[:, None, :, :]
                dr = (x ** 3 * coefs[:, :, 0, :] + y ** 3 * coefs[:, :, 1, :] + x ** 2 * y * coefs[:, :, 2, :]
                      + y ** 2 * x * coefs[:, :, 3, :] + x ** 4 * coefs[:, :, 4, :] + y ** 4 * coefs[:, :, 5, :]
                      + y ** 3 * x * coefs[:, :, 6, :] + x ** 2 * y ** 2 * coefs[:, :, 7, :]
                      + y * x ** 3 * coefs[:, :, 8, :]
                      )
                # shape^: batch_size, spatial, 3
                r_new = r_new + dr

                coefs = self.train_var_dict['u_higher_order']
                coefs = self._tf_gather_if_necessary(coefs, batch_inds)[:, None, :, :]
                du = (x ** 3 * coefs[:, :, 0, :] + y ** 3 * coefs[:, :, 1, :] + x ** 2 * y * coefs[:, :, 2, :]
                      + y ** 2 * x * coefs[:, :, 3, :] + x ** 4 * coefs[:, :, 4, :] + y ** 4 * coefs[:, :, 5, :]
                      + y ** 3 * x * coefs[:, :, 6, :] + x ** 2 * y ** 2 * coefs[:, :, 7, :]
                      + y * x ** 3 * coefs[:, :, 8, :]
                      )
                # shape^: batch_size, spatial, 3
                u_new = u_new + du

            u_new, _ = tf.linalg.normalize(u_new, axis=-1)  # renormalize to unit vector
            r_new = tf.reshape(r_new, [-1, 3])
            u_new = tf.reshape(u_new, [-1, 3])

            return r_new, u_new
        else:
            return r_before_sample, u_before_sample

    def _propagate_thru_nmr_tube(self, r, u):
        # modeling nmr tube as a hemisphere attached to a cylinder with inner and outer radii. Pose of the tube is
        # handled by rotating/shifting the boundary conditions. This operation is applied after nonparametric stuff.
        # r and u are of shape _ x 3.

        # unpack parameters:
        r_outer = self.train_var_dict['nmr_outer_radius'] * 1000  # convert to um
        r_inner = self.train_var_dict['nmr_inner_radius'] * 1000
        normal = self.train_var_dict['nmr_normal']
        theta = self.train_var_dict['nmr_theta']
        delta_r = self.train_var_dict['nmr_delta_r'] * 1000
        n_glass = self.train_var_dict['n_glass']
        n_medium = self.train_var_dict['n_medium']

        # first, rotate and translate the input rays to simulate a rotated nmr tube:
        rotmat = self._axis_angle_rotmat(normal, theta)
        r_ = r @ rotmat + delta_r[None, :]  # shape: _, 3
        u_ = u @ rotmat

        # next, intersect with both spherical and cylindrical outer surfaces:
        r_sphere, d_to_sphere, discriminant_sphere = self._propagate_to_sphere(r_, u_, r_outer)
        r_cylinder, d_to_cylinder, discriminant_cylinder = self._propagate_to_cylinder(r_, u_, r_outer)

        # next, for each ray, pick out either the spherical or cylindrical cases:
        use_sphere = r_cylinder[:, 2:3] < 0
        r_first_surface = tf.where(use_sphere, r_sphere, r_cylinder)  # pick out spherical or cylindrical case;
        self.discriminant_1st_surf = tf.where(use_sphere, discriminant_sphere, discriminant_cylinder)
        self.missed_1st_surf = self.discriminant_1st_surf < 0
        r_first_surface = tf.where(self.missed_1st_surf, r_, r_first_surface)  # if it missed the surface, then retain previous ray
        d_to_first_surface = tf.where(use_sphere, d_to_sphere, d_to_cylinder)
        d_to_first_surface = tf.where(self.missed_1st_surf, tf.zeros_like(d_to_first_surface), d_to_first_surface)  # if missed, then
        # we don't propagate

        # refract using Snell's law to get new direction:
        n_sphere, _ = tf.linalg.normalize(r_sphere, axis=1)  # surface normal in case of sphere
        n_cylinder, _ = tf.linalg.normalize(tf.concat([r_cylinder[:, 0:1],  # surface normal in case of cylinder
                                                       r_cylinder[:, 1:2],
                                                       tf.zeros_like(r_cylinder[:, 1:2])], axis=1), axis=1)
        n_first_surface = tf.where(use_sphere, n_sphere, n_cylinder)  # pick out which normal
        u_first_surface, refract_error = self._refract_snell(u_, n_first_surface, 1 / n_glass)
        self.missed_1st_surf_or_refract_error = tf.math.logical_or(refract_error,
                                                                   self.missed_1st_surf)  # if the surface was missed, or refraction error
        u_first_surface = tf.where(self.missed_1st_surf_or_refract_error, u_, u_first_surface)  # if it missed the surface, then this refraction event
        # is meaningless, so retrain old direction

        # propagate to next surface (the inner surface), repeating the above:
        r_sphere, d_to_sphere, discriminant_sphere = self._propagate_to_sphere(r_first_surface, u_first_surface,
                                                                               r_inner)
        r_cylinder, d_to_cylinder, discriminant_cylinder = self._propagate_to_cylinder(r_first_surface, u_first_surface,
                                                                                       r_inner)
        use_sphere = r_cylinder[:, 2:3] < 0
        r_second_surface = tf.where(use_sphere, r_sphere, r_cylinder)  # pick out spherical or cylindrical case
        self.discriminant_2nd_surf = tf.where(use_sphere, discriminant_sphere, discriminant_cylinder)
        self.missed_2nd_surf = self.discriminant_2nd_surf < 0
        r_second_surface = tf.where(self.missed_2nd_surf, r_first_surface, r_second_surface)  # if it missed the surface, then retain
        # previous ray
        d_to_second_surface = tf.where(use_sphere, d_to_sphere, d_to_cylinder)
        d_to_second_surface = tf.where(self.missed_2nd_surf, tf.zeros_like(d_to_second_surface), d_to_second_surface)  # if missed,
        # then we don't propagate
        n_sphere, _ = tf.linalg.normalize(r_sphere, axis=1)  # surface normal in case of sphere
        n_cylinder, _ = tf.linalg.normalize(tf.concat([r_cylinder[:, 0:1],  # surface normal in case of cylinder
                                                       r_cylinder[:, 1:2],
                                                       tf.zeros_like(r_cylinder[:, 1:2])], axis=1), axis=1)
        n_second_surface = tf.where(use_sphere, n_sphere, n_cylinder)  # pick out which normal
        u_second_surface, refract_error = self._refract_snell(u_first_surface, n_second_surface, n_glass / n_medium)
        self.missed_2nd_surf_or_refract_error = tf.math.logical_or(refract_error, self.missed_2nd_surf)  # if the surface was missed, or refraction error
        u_second_surface = tf.where(self.missed_2nd_surf_or_refract_error, u_first_surface, u_second_surface)  # if it missed the surface, then this
        # refraction event is meaningless, so retrain old direction

        # now, we want to go back to roughly the origin, so we propagate from the second surface the negative distance
        # that we've propagated so far:
        u_new = u_second_surface  # this doesn't change
        r_new = r_second_surface + u_new * -(d_to_first_surface + d_to_second_surface)

        # monitor for debugging:
        self.tensors_to_track['r_first_surface'] = r_first_surface
        self.tensors_to_track['r_second_surface'] = r_second_surface
        self.tensors_to_track['r_end'] = r_new
        self.tensors_to_track['r_before_nmr'] = r
        self.tensors_to_track['u_first_surface'] = u_first_surface
        self.tensors_to_track['u_second_surface'] = u_second_surface
        self.tensors_to_track['u_end'] = u_new
        self.tensors_to_track['u_before_nmr'] = u

        self.tensors_to_track['missed_1st_surf'] = self.missed_1st_surf
        self.tensors_to_track['missed_2nd_surf'] = self.missed_2nd_surf
        self.tensors_to_track['missed_1st_surf_or_refract_error'] = self.missed_1st_surf_or_refract_error
        self.tensors_to_track['missed_2nd_surf_or_refract_error'] = self.missed_2nd_surf_or_refract_error
        self.tensors_to_track['discriminant_1st_surf'] = self.discriminant_1st_surf
        self.tensors_to_track['discriminant_2nd_surf'] = self.discriminant_2nd_surf

        return r_new, u_new

    def _refract_snell(self, u, normals, ri_ratio):
        # determine new ray directions given input directions, u, and the corresponding surface normals.
        # ri_ratio: ri_incident/ri_excident
        _n_dot_u = - tf.einsum('ij,ij->i', normals, u)  # negative n dot u
        sqrt_arg = 1 - ri_ratio ** 2 * (1 - _n_dot_u ** 2)
        refract_error = sqrt_arg[:, None] < 0  # e.g., TIR
        u_out = ri_ratio * u + (ri_ratio * _n_dot_u -
                                tf.sqrt(tf.maximum(tf.abs(sqrt_arg), 1e-7)))[:, None] * normals
        # take the abs value to avoid nan, which tf.where doesn't like
        return u_out, refract_error

    def _propagate_to_sphere(self, r, u, radius):
        # sphere is centered at the origin
        u_dot_r = tf.einsum('ij,ij->i', u, r)
        discriminant = radius ** 2 - tf.norm(r, axis=1) ** 2 + u_dot_r ** 2
        # missed = discriminant[:, None] < 0  # missed the sphere; will yield nan in the next line:
        dist_to_sphere = - u_dot_r - tf.sqrt(tf.maximum(tf.abs(discriminant), 1e-7))  # take neg sol
        # take the abs value to avoid nan, which tf.where doesn't like
        dist_to_sphere = dist_to_sphere[:, None]
        r_at_sphere = r + u * dist_to_sphere  # position at dome
        return r_at_sphere, dist_to_sphere, discriminant[:, None]
        # ^discriminant can help with bulk tube registration

    def _propagate_to_cylinder(self, r, u, radius):
        # cylinder is centered at the origin about the z axis
        r_x = r[:, 0:1]
        r_y = r[:, 1:2]
        u_x = u[:, 0:1]
        u_y = u[:, 1:2]
        uxy_dot_rxy = r_x * u_x + r_y * u_y
        ux2_uy2 = u_x ** 2 + u_y ** 2
        discriminant = radius ** 2 * ux2_uy2 - (u_y * r_x - u_x * r_y) ** 2
        # missed = discriminant < 0  # missed the cylinder; will yield nan in the next line:
        dist_to_cylinder = -(uxy_dot_rxy + tf.sqrt(tf.maximum(tf.abs(discriminant), 1e-7))) / ux2_uy2
        # take the abs value to avoid nan, which tf.where doesn't like
        r_at_cylinder = r + u * dist_to_cylinder
        return r_at_cylinder, dist_to_cylinder, discriminant

    def _extra_finetune_rays(self, r_before_sample, u_before_sample, batch):
        # this is applied AFTER refraction through the tube, and can be an arbitrary polynomial order, specified via
        # self.extra_finetune_order.
        # batch needs to be provided so we can get the lateral coordinates of the rays that were propagated (since the
        # polynomials are of these 2D coordinates), so we have to copy some code from the beginning of
        # _propagate_to_parabolic_focus:

        # gather the relevant rays to propagate, corresponding to the batch:
        if self.batch_size is None:
            # batch should be None; or even if not, it's ignored
            num_images = self.num_images
            num_pixels = np.prod(self.im_downsampled_shape)
            batch_inds = None
            galvo_xy_batch_ = self.galvo_xy_downsamp
        else:
            if self.batch_across_images:
                num_images = self.batch_size
                num_pixels = np.prod(self.im_downsampled_shape)
                batch_inds = batch  # interpret input to this function as batch inds
                galvo_xy_batch_ = self.galvo_xy_downsamp
            else:
                num_images = self.num_images
                num_pixels = self.batch_size
                batch_inds = None
                galvo_xy_batch_ = batch  # shape: num_pixels, 2
                # ^interpret input to this function as galvo_xy_batch (a random subset of pixels coordinates)

        r = tf.reshape(r_before_sample, [num_images, num_pixels, 3])  # unflatten
        u = tf.reshape(u_before_sample, [num_images, num_pixels, 3])  # unflatten

        xy_p = galvo_xy_batch_  # shape: spatial, 2
        x_p = xy_p[:, 0]  # shape: spatial,
        y_p = xy_p[:, 1]

        polynomials = (tf.math.pow(x_p[None, :, None], self.extra_orders[None, None, :, 0]) *
                       tf.math.pow(y_p[None, :, None], self.extra_orders[None, None, :, 1]))
        # shape^: 1, spatial, num_terms (no coefficients yet)

        # apply coefficients for spatial adjustment:
        coefs_r = self.train_var_dict['r_extra']  # shape: num_images, num_terms, 3;  different poly for x,y,z
        polynomials_with_coefs_r = polynomials[:, :, :, None] * coefs_r[:, None, :, :]
        # shape^: num_images, spatial, num_terms, 3
        delta_r = tf.reduce_sum(polynomials_with_coefs_r, axis=2)  # sum up all polynomial terms to yield
        # final result; shape^: num_images, spatial, 3

        # apply coefficients for angular adjustment:
        coefs_u = self.train_var_dict['u_extra']  # shape: num_images, num_terms, 3;  different poly for x,y,z
        polynomials_with_coefs_u = polynomials[:, :, :, None] * coefs_u[:, None, :, :]
        # shape^: num_images, spatial, num_terms, 3
        delta_u = tf.reduce_sum(polynomials_with_coefs_u, axis=2)  # sum up all polynomial terms to yield
        # final result; shape^: num_images, spatial, 3

        # adjust BCs:
        r_new = r + delta_r
        u_new = u + delta_u

        u_new, _ = tf.linalg.normalize(u_new, axis=-1)  # renormalize to unit vector
        r_new = tf.reshape(r_new, [-1, 3])
        u_new = tf.reshape(u_new, [-1, 3])

        return r_new, u_new

    def _propagate_rays_thru_sample(self, r_at_sample, u_at_sample):
        # get straight line from position and direction

        # scaled np.arange does the straightline rayprop through homogeneous medium:
        line = tf.range(self.num_z_steps, dtype=tf.float32)
        line -= tf.reduce_mean(line)
        if self.dither_rays:  # rays are dithered in the same way for all rays
            dither = tf.random.uniform(shape=[self.num_z_steps], minval=-0.5, maxval=0.5, dtype=tf.float32)
            line = line + dither

        self.linear_path = self.step / self.scale * (line + 1)
        self.propped = r_at_sample[:, None, :] + self.linear_path[None, :, None] * u_at_sample[:, None, :]
        # shape: batch_size * im x * im y, numz, 3

        # unpack for convenience:
        self.x_path, self.y_path, self.z_path = [unpacked[..., 0] for unpacked in tf.split(self.propped, 3, axis=2)]

    def _propagate(self, batch_inds):
        # this function is a convenience function that bundles all of the propagate function calls in one

        r_before_sample, u_before_sample = self._propagate_to_parabolic_focus(batch_inds)

        if 'nmr' in self.propagation_model:
            r_before_sample, u_before_sample = self._propagate_thru_nmr_tube(r_before_sample, u_before_sample)

        if 'extra_finetune' in self.propagation_model:
            r_before_sample, u_before_sample = self._extra_finetune_rays(r_before_sample, u_before_sample, batch_inds)

        # global shift:
        r_before_sample = r_before_sample + self.xyz_offset[None, :]

        return r_before_sample, u_before_sample

    def _forward_predict(self, im_downsamp_batch=None, batch_inds=None, dataset_index=None, weight_background=.003):
        # im_downsamp_batch shape: num_images, flattened spatial, num_channels

        if self.batch_size is None:
            # batch should be None; or even if not, it's ignored
            num_images = self.num_images
            num_pixels = np.prod(self.im_downsampled_shape)
            batch_inds = None
            im_downsamp_batch = self.stack_downsamp  # the input im_downsamp_batch is None in this case
        else:
            if self.batch_across_images:
                num_images = self.batch_size
                num_pixels = np.prod(self.im_downsampled_shape)
            else:
                num_images = self.num_images
                num_pixels = self.batch_size

        self.tensors_to_track['x_path'] = self.x_path
        self.tensors_to_track['y_path'] = self.y_path
        self.tensors_to_track['z_path'] = self.z_path  # shape: num_images * spatial, numz

        im_downsamp_batch = tf.cast(im_downsamp_batch, dtype=tf.float32) / 255  # from uint8
        im = im_downsamp_batch

        # broadcast to include z dimension:
        im = tf.broadcast_to(im[:, :, None, :],
                             (num_images, num_pixels, self.num_z_steps, self.num_channels))
        # flatten image pixel (except channels):
        im = tf.reshape(im, (-1, self.num_channels))  # shape: num_images * spatial(xyz), num channels

        # convert to pixel units from physical spatial units:
        recon_size_dims = tf.cast(self.recon_shape, tf.float32)

        x_float = ((self.x_path / self.recon_fov[0]) + .5) * recon_size_dims[0]
        y_float = ((self.y_path / self.recon_fov[1]) + .5) * recon_size_dims[1]
        z_float = ((self.z_path / self.recon_fov[2]) + .5) * recon_size_dims[2]

        if self.occupancy_grid is not None:
            # remove points that are unoccupied

            # need to expose all dimensions, which can be ragged:
            x_float = tf.reshape(x_float, (num_images, num_pixels, self.num_z_steps))
            y_float = tf.reshape(y_float, (num_images, num_pixels, self.num_z_steps))
            z_float = tf.reshape(z_float, (num_images, num_pixels, self.num_z_steps))

            if dataset_index is not None:
                occupancy_grid_i = self.occupancy_grid[dataset_index]
            else:
                occupancy_grid_i = self.occupancy_grid[0]  # only one present
            occ_shape = occupancy_grid_i.shape
            x_occ = tf.cast(tf.round(x_float * occ_shape[0] / recon_size_dims[0]), dtype=tf.int32)
            y_occ = tf.cast(tf.round(y_float * occ_shape[1] / recon_size_dims[1]), dtype=tf.int32)
            z_occ = tf.cast(tf.round(z_float * occ_shape[2] / recon_size_dims[2]), dtype=tf.int32)
            x_occ = tf.clip_by_value(x_occ, 0, occ_shape[0] - 1)
            y_occ = tf.clip_by_value(y_occ, 0, occ_shape[1] - 1)
            z_occ = tf.clip_by_value(z_occ, 0, occ_shape[2] - 1)
            xyz_occ = tf.stack([x_occ, y_occ, z_occ], axis=-1)
            ray_occupancy = tf.gather_nd(occupancy_grid_i, xyz_occ)  # did each pixel of the rays visit something?
            ray_occupancy = tf.concat([tf.ones_like(ray_occupancy[:, :, 0:1]), ray_occupancy[:, :, 1:]], axis=-1)
            # ^ this forces there to be at least one ray in each ray
            ray_occupancy = tf.stop_gradient(ray_occupancy)

            # keep only the pixels that visited part of the object:
            x_float = tf.ragged.boolean_mask(x_float, ray_occupancy)
            y_float = tf.ragged.boolean_mask(y_float, ray_occupancy)
            z_float = tf.ragged.boolean_mask(z_float, ray_occupancy)
            xyz_nested_row_splits = x_float.nested_row_splits  # keep a copy of this for reshaping forward pred later

            # flatten:
            x_float = x_float.flat_values
            y_float = y_float.flat_values
            z_float = z_float.flat_values

        else:
            x_float = tf.reshape(x_float, [-1])
            y_float = tf.reshape(y_float, [-1])
            z_float = tf.reshape(z_float, [-1])

        if self.interp_rays:
            # rather than remove points that don't fall in FOV, simply use clip_by_value (to preserve shape):
            x_float = tf.clip_by_value(x_float, 0, recon_size_dims[0])
            y_float = tf.clip_by_value(y_float, 0, recon_size_dims[1])
            z_float = tf.clip_by_value(z_float, 0, recon_size_dims[2])

            # trilinear interp (for scattering and gathering):
            x_floor = tf.floor(x_float)
            x_ceil = x_floor + 1
            z_floor = tf.floor(z_float)
            z_ceil = z_floor + 1
            y_floor = tf.floor(y_float)
            y_ceil = y_floor + 1

            fx = x_float - x_floor
            cx = x_ceil - x_float
            fz = z_float - z_floor
            cz = z_ceil - z_float
            fy = y_float - y_floor
            cy = y_ceil - y_float

            # cast into integers:
            x_floor = tf.cast(x_floor, dtype=tf.int32)
            x_ceil = tf.cast(x_ceil, dtype=tf.int32)
            z_floor = tf.cast(z_floor, dtype=tf.int32)
            z_ceil = tf.cast(z_ceil, dtype=tf.int32)
            y_floor = tf.cast(y_floor, dtype=tf.int32)
            y_ceil = tf.cast(y_ceil, dtype=tf.int32)

            # generate the coordinates of the projection cells:
            xyzfff = tf.stack([x_floor, y_floor, z_floor], 1)
            xyzfcf = tf.stack([x_floor, y_ceil, z_floor], 1)
            xyzcff = tf.stack([x_ceil, y_floor, z_floor], 1)
            xyzccf = tf.stack([x_ceil, y_ceil, z_floor], 1)
            xyzffc = tf.stack([x_floor, y_floor, z_ceil], 1)
            xyzfcc = tf.stack([x_floor, y_ceil, z_ceil], 1)
            xyzcfc = tf.stack([x_ceil, y_floor, z_ceil], 1)
            xyzccc = tf.stack([x_ceil, y_ceil, z_ceil], 1)

            # gaussian-weighted factors (these are for interp_project and for the gathering stage after projection):
            fx = tf.exp(-fx ** 2 / 2. / self.sig_proj ** 2)
            fy = tf.exp(-fy ** 2 / 2. / self.sig_proj ** 2)
            fz = tf.exp(-fz ** 2 / 2. / self.sig_proj ** 2)
            cx = tf.exp(-cx ** 2 / 2. / self.sig_proj ** 2)
            cy = tf.exp(-cy ** 2 / 2. / self.sig_proj ** 2)
            cz = tf.exp(-cz ** 2 / 2. / self.sig_proj ** 2)

            # reconstruct:
            # compute the interpolated normalize tensor here:
            # _8 is used because for 3D, trilinear interpolation uses 8 cubes
            xyz_8 = tf.concat([xyzfff, xyzfcf, xyzcff, xyzccf, xyzffc, xyzfcc, xyzcfc, xyzccc], 0)

            # compute the interpolated backprojection:
            # it might be more efficient to use broadcasting for this:
            f1 = fx * fy * fz
            f2 = fx * cy * fz
            f3 = cx * fy * fz
            f4 = cx * cy * fz
            f5 = fx * fy * cz
            f6 = fx * cy * cz
            f7 = cx * fy * cz
            f8 = cx * cy * cz

        self.recon = self.train_var_dict['recon']

        # forward prediction:
        if self.interp_rays:
            # gathering stage for computing the loss
            fff = tf.gather_nd(self.recon, xyzfff)
            fcf = tf.gather_nd(self.recon, xyzfcf)
            cff = tf.gather_nd(self.recon, xyzcff)
            ccf = tf.gather_nd(self.recon, xyzccf)
            ffc = tf.gather_nd(self.recon, xyzffc)
            fcc = tf.gather_nd(self.recon, xyzfcc)
            cfc = tf.gather_nd(self.recon, xyzcfc)
            ccc = tf.gather_nd(self.recon, xyzccc)

            forward = (ccc * f8[:, None] +
                       ccf * f4[:, None] +
                       cff * f3[:, None] +
                       cfc * f7[:, None] +
                       fcc * f6[:, None] +
                       fcf * f2[:, None] +
                       fff * f1[:, None] +
                       ffc * f5[:, None])

            forward /= (f8 +
                        f4 +
                        f3 +
                        f7 +
                        f6 +
                        f2 +
                        f1 +
                        f5)[:, None]
        else:
            # clip out values that go out of bounds:
            x_float = tf.clip_by_value(x_float, 0, recon_size_dims[0])
            y_float = tf.clip_by_value(y_float, 0, recon_size_dims[1])
            z_float = tf.clip_by_value(z_float, 0, recon_size_dims[2])

            xyz = tf.stack([x_float, y_float, z_float], axis=1)  # _ by 3

            xyz = tf.cast(tf.math.round(xyz), dtype=tf.int32)
            forward = tf.gather_nd(self.recon, xyz)

        # project across axial:
        # unflatten to expose num_z dimension:
        if self.occupancy_grid is None:
            forward = tf.reshape(forward, (num_images, num_pixels, self.num_z_steps, self.num_channels))
        else:
            forward = tf.RaggedTensor.from_nested_row_splits(forward, xyz_nested_row_splits)

        # save this for computing regularization terms based on the current batch:
        self.forward_recon = forward

        if self.z_int_profile is not None:
            if self.occupancy_grid is None:
                forward = forward * self.z_int_profile[None, None, :, None]
            else:
                z_int_profile_masked = tf.ragged.boolean_mask(
                    tf.broadcast_to(self.z_int_profile[None, None, :, None],
                                    (num_images, num_pixels, self.num_z_steps, self.num_channels)), ray_occupancy)
                # for some reason, the tf.map_fn version of this was very slow, so I broadcasted instead ...
                forward = forward * z_int_profile_masked

        if self.use_attenuation_map:
            # do the same thing as we did for recon (but no channels dimension):
            atten = self.train_var_dict['attenuation_map']
            self.attenuation_map = atten

            if self.interp_rays:
                fff = tf.gather_nd(atten, xyzfff)
                fcf = tf.gather_nd(atten, xyzfcf)
                cff = tf.gather_nd(atten, xyzcff)
                ccf = tf.gather_nd(atten, xyzccf)
                ffc = tf.gather_nd(atten, xyzffc)
                fcc = tf.gather_nd(atten, xyzfcc)
                cfc = tf.gather_nd(atten, xyzcfc)
                ccc = tf.gather_nd(atten, xyzccc)

                forward_atten = (ccc * f8 +
                                 ccf * f4 +
                                 cff * f3 +
                                 cfc * f7 +
                                 fcc * f6 +
                                 fcf * f2 +
                                 fff * f1 +
                                 ffc * f5)

                forward_atten /= (f8 +  # should reuse this from above ...
                                  f4 +
                                  f3 +
                                  f7 +
                                  f6 +
                                  f2 +
                                  f1 +
                                  f5)
            else:
                forward_atten = tf.gather_nd(atten, xyz)

            if self.occupancy_grid is None:
                forward_atten = tf.reshape(forward_atten, (num_images, num_pixels, self.num_z_steps))
            else:
                forward_atten = tf.RaggedTensor.from_nested_row_splits(forward_atten, xyz_nested_row_splits)

            # save this for computing regularization terms based on the current batch:
            self.forward_atten = forward_atten

            if self.z_int_profile is not None:
                if self.occupancy_grid is None:
                    forward_atten = forward_atten * self.z_int_profile[None, None, :]
                else:
                    forward_atten = forward_atten * z_int_profile_masked[..., 0]

            cumsum = tf.math.cumsum(forward_atten, axis=2)  # for computing beer-lambert law
            forward = forward * tf.exp(-cumsum)[:, :, :, None]

            forward = tf.reduce_mean(forward, axis=2)  # project across axial dimension

        # apply scale and bias to forward prediction:
        per_image_scale = self.train_var_dict['per_image_scale'] ** 2  # square to keep positive
        per_image_bias = self.train_var_dict['per_image_bias']
        # per_image_scale_norm = per_image_scale / tf.math.reduce_prod(per_image_scale) ** (1/self.num_images)
        # normalize by geometric mean
        per_image_scale_norm = per_image_scale / tf.reduce_mean(per_image_scale)  # geometric mean in principle makes
        # more sense, but could lead to overflow errors; another option would be to take the power inside the product
        # when computing the geo mean, but could result in loss in precision...
        per_image_bias_norm = per_image_bias - tf.reduce_mean(per_image_bias)
        scale_batch = self._tf_gather_if_necessary(per_image_scale_norm, batch_inds)
        bias_batch = self._tf_gather_if_necessary(per_image_bias_norm, batch_inds)
        forward = forward * scale_batch[:, None, None] + bias_batch[:, None, None]
        self.tensors_to_track['forward'] = forward

        self.error = im_downsamp_batch - forward
        if weight_background is None:
            self.MSE = tf.reduce_mean(self.error ** 2)
        else:
            weight = 1 / (im_downsamp_batch + weight_background)  # remember that im_downsamp_batch is the original RGB/255
            # the constant in the denominator controls how much weight to give the background pixels
            weight_sum = tf.reduce_sum(weight)
            self.MSE = tf.reduce_sum(self.error ** 2 * weight) / weight_sum

    def _add_regularization_loss(self, reg_coefs):
        # create loss_list of all the loss terms

        # always have data-dependent loss:
        self.loss_list = [self.MSE]
        self.loss_list_names = ['MSE']

        if 'TV' in reg_coefs:
            loss = self._TV_loss()
            self.loss_list.append(reg_coefs['TV'] * loss)
            self.loss_list_names.append('TV')
        if 'TV2' in reg_coefs:
            loss = self._TV2_loss()
            self.loss_list.append(reg_coefs['TV2'] * loss)
            self.loss_list_names.append('TV2')
        if 'L1' in reg_coefs:
            loss = self._L1_loss()
            self.loss_list.append(reg_coefs['L1'] * loss)
            self.loss_list_names.append('L1')
        if 'L2' in reg_coefs:
            loss = self._L2_loss()
            self.loss_list.append(reg_coefs['L2'] * loss)
            self.loss_list_names.append('L2')
        if 'bkgd_seg' in reg_coefs:
            self.loss_list.append(reg_coefs['bkgd_seg'] * self.background_loss)
            self.loss_list_names.append('bkgd_seg')
        if 'TV_attenuation' in reg_coefs:
            loss = self._TV_loss(target='attenuation_map')
            self.loss_list.append(reg_coefs['TV_attenuation'] * loss)
            self.loss_list_names.append('TV_attenuation')
        if 'TV2_attenuation' in reg_coefs:
            loss = self._TV2_loss(target='attenuation_map')
            self.loss_list.append(reg_coefs['TV2_attenuation'] * loss)
            self.loss_list_names.append('TV2_attenuation')
        if 'L1_attenuation' in reg_coefs:
            loss = self._L1_loss(target='attenuation_map')
            self.loss_list.append(reg_coefs['L1_attenuation'] * loss)
            self.loss_list_names.append('L1_attenuation')
        if 'L2_attenuation' in reg_coefs:
            loss = self._L2_loss(target='attenuation_map')
            self.loss_list.append(reg_coefs['L2_attenuation'] * loss)
            self.loss_list_names.append('L2_attenuation')
        if 'L1_attenuation_batch' in reg_coefs:
            scale = reg_coefs['L1_attenuation_batch']['scale']
            coef = reg_coefs['L1_attenuation_batch']['coef']
            loss = self._L1_attenuation_batch(scale)
            self.loss_list.append(coef * loss)
            self.loss_list_names.append('L1_attenuation_batch')
        if 'L2_attenuation_batch' in reg_coefs:
            scale = reg_coefs['L2_attenuation_batch']['scale']
            coef = reg_coefs['L2_attenuation_batch']['coef']
            loss = self._L2_attenuation_batch(scale)
            self.loss_list.append(coef * loss)
            self.loss_list_names.append('L2_attenuation_batch')

    def _L2_attenuation_batch(self, scale):
        atten_proj = tf.reduce_mean(self.forward_atten ** 2, axis=2)
        recon_proj = tf.reduce_mean(tf.stop_gradient(self.forward_recon), axis=(2, 3))  # also remove channel dimension.
        # For weighting the attenuation map projection L1 regularization. Don't use the raw image pixels, because a dark
        # region may be due to attenuation!
        # shapes of the above: num_images, num_pixels
        return tf.reduce_mean(1 / (scale * recon_proj + 1) * atten_proj)
    def _L1_attenuation_batch(self, scale):
        atten_proj = tf.reduce_mean(tf.abs(self.forward_atten), axis=2)
        recon_proj = tf.reduce_mean(tf.stop_gradient(self.forward_recon), axis=(2, 3))  # also remove channel dimension.
        # For weighting the attenuation map projection L1 regularization. Don't use the raw image pixels, because a dark
        # region may be due to attenuation!
        # shapes of the above: num_images, num_pixels
        return tf.reduce_mean(1 / (scale * recon_proj + 1) * atten_proj)

    def _TV_loss(self, target='recon'):
        # total variation
        if target == 'recon':
            R = self.recon
        elif target == 'attenuation_map':
            R = self.attenuation_map
        R_ = R[:-1, :-1, :-1]
        d0 = R[1:, :-1, :-1] - R_
        d1 = R[:-1, 1:, :-1] - R_
        d2 = R[:-1, :-1, 1:] - R_
        return tf.reduce_sum(tf.sqrt(d0 ** 2 + d1 ** 2 + d2 ** 2 + 1e-7))

    def _TV2_loss(self, target='recon'):
        # total variation squared
        if target == 'recon':
            R = self.recon
        elif target == 'attenuation_map':
            R = self.attenuation_map
        R_ = R[:-1, :-1, :-1]
        d0 = R[1:, :-1, :-1] - R_
        d1 = R[:-1, 1:, :-1] - R_
        d2 = R[:-1, :-1, 1:] - R_
        return tf.reduce_sum(d0 ** 2 + d1 ** 2 + d2 ** 2)

    def _L1_loss(self, target='recon'):
        if target == 'recon':
            R = self.recon
        elif target == 'attenuation_map':
            R = self.attenuation_map
        return tf.reduce_sum(tf.sqrt(R ** 2 + 1e-7))

    def _L2_loss(self, target='recon'):
        if target == 'recon':
            R = self.recon
        elif target == 'attenuation_map':
            R = self.attenuation_map
        return tf.reduce_sum(R ** 2)


    @tf.function
    def gradient_update(self, batch=None, update_gradient=True, reg_coefs=None,
                        return_tracked_tensors=False,
                        return_grads=False, return_loss_only=False, dataset_index=None,
                        weight_background=None
                        ):
        # batch: returned by the tf.dataset and will be unpacked here.
        # reg_coefs: dictionary of regularization coefficients.
        # return_tracked_tensors: if True, will return tracked tensors from tf graph (that are not tf.Variables).
        # dataset_index: when optimizing xyz_shifts (i.e., when using datasets of multiple shifted versions of the
        # same object), this specifies which dataset we're using.
        # weight_background: give more weight to the background pixels when computing loss.

        if self.batch_size is None:
            # batch is None in this case
            backproject_input = None
            propagate_input = None
        else:
            if self.batch_across_images:
                backproject_input = batch[0]  # im_downsamp
                propagate_input = batch[1]  # batch_inds
                batch_inds = propagate_input
            else:
                backproject_input = tf.transpose(batch[0], (1, 0, 2))  # im_downsamp; move spatial dim to dim 1
                propagate_input = batch[1]  # galvo_xy; no need to transpose, because no camera/probe dimension
                batch_inds = batch[2]  # batch_inds
                if self.yield_identifier and dataset_index is None:  # if the user supplies dataset_index to this
                    # function, then you override the one provided by the batch
                    dataset_index = batch[3]
                    dataset_index = dataset_index[0]  # a little wasteful to yield a vector of identical values

        with tf.GradientTape() as tape:
            r_before_sample, u_before_sample = self._propagate(propagate_input)
            self._propagate_rays_thru_sample(r_before_sample, u_before_sample)
            self._forward_predict(backproject_input, batch_inds=batch_inds, dataset_index=dataset_index,
                                  weight_background=weight_background)

            if reg_coefs is not None:
                self._add_regularization_loss(reg_coefs)
                loss = tf.reduce_sum(self.loss_list)
            else:
                loss = self.MSE

        trainables = [(var, optim) for var, optim, train in  # avoid computing gradients when not used
                      zip(self.train_var_dict.values(),
                          self.optimizer_dict.values(),
                          self.trainable_or_not) if train]
        var_list = [pair[0] for pair in trainables]
        grads = tape.gradient(loss, var_list)


        # apply gradient update for each optimizer:
        # note that this assumes that the dictionaries are ordered!
        if update_gradient:
            for grad, (var, optimizer) in zip(grads, trainables):
                optimizer.apply_gradients([(grad, var)])

        if self.force_positive_recon:
            self.train_var_dict['recon'].assign(tf.maximum(self.train_var_dict['recon'], 0))
            if self.use_attenuation_map:
                self.train_var_dict['attenuation_map'].assign(tf.maximum(self.train_var_dict['attenuation_map'], 0))

                # zero the borders:
                zero_slice = np.zeros(self.recon_shape[0:2])
                self.train_var_dict['attenuation_map'][0].assign(zero_slice)
                self.train_var_dict['attenuation_map'][-1].assign(zero_slice)
                self.train_var_dict['attenuation_map'][:, 0].assign(zero_slice)
                self.train_var_dict['attenuation_map'][:, -1].assign(zero_slice)
                self.train_var_dict['attenuation_map'][:, :, 0].assign(zero_slice)
                self.train_var_dict['attenuation_map'][:, :, -1].assign(zero_slice)

        if return_loss_only:
            if reg_coefs is not None:
                return_list = [self.loss_list]
            else:
                return_list = [self.MSE]
        else:
            if reg_coefs is not None:
                return_list = [self.loss_list, self.recon]
            else:
                return_list = [self.MSE, self.recon]

            if return_tracked_tensors:
                return_list.append(self.tensors_to_track)
            if return_grads:
                return_list.append(grads)

        return return_list

    def generate_full_forward_prediction(self, dataset, batch_size, max_iter=None):
        # generate a full forward prediction for all pixels of all cameras in the array (chosen with inds_keep)

        forward_prediction = np.zeros(self.stack.shape[:-1], dtype=np.float32)
        jj = 0
        if max_iter is None:
            one_epoch = int(np.prod(self.stack.shape[1:]) / batch_size)
        else:
            one_epoch = max_iter
        for batch in tqdm(dataset, total=one_epoch):

            # this generates the forward prediction for a batch, stored in tracked:
            _, _, tracked = self.gradient_update(batch, update_gradient=False, return_tracked_tensors=True)

            # assign:
            batch_inds = np.asarray(batch[2])
            r, c = np.unravel_index(batch_inds, self.stack.shape[1:3])
            pred = tracked['forward'].numpy()
            for i in range(len(self.inds_keep)):
                forward_prediction[i, r, c] = pred[i, :, 0]

            if jj > one_epoch:
                break
            else:
                jj += 1

        return forward_prediction


def process_data(filepath, blur_sigma=None, intensity_threshold=None):
    # load from nc filepath, debayer, take green channel, and optionally blur or threshold the images

    data = xr.open_dataset(filepath)
    if 'mcam_data' in data:
        mcam_data = data.mcam_data
    else:
        mcam_data = data.images
    nc_shape = mcam_data.shape
    stack = np.asarray(mcam_data)

    stack = stack.reshape((-1,) + stack.shape[len(nc_shape) - 2:])  # flatten camera dimensions, handling video/nonvideo
    stack = np.stack([cv2.cvtColor(np.asarray(im), cv2.COLOR_BAYER_GB2RGB)[..., 1:2] for im in stack])  # debayer

    print(stack.shape)

    if blur_sigma is not None:
        stack = stack[:, :, :, 0]  # to facilitate faster indexing
        new_stack = np.empty_like(stack)
        for i, im in tqdm(enumerate(stack)):
            new_stack[i] = scipy.ndimage.gaussian_filter(im, blur_sigma, truncate=2)
        stack = new_stack[:, :, :, None]  # add back this dimension

    if intensity_threshold is not None:
        for i, im in enumerate(stack):
            bw = im > intensity_threshold
            labeled = measure.label(bw)
            props = measure.regionprops(labeled)

            # get max size
            size = max([i.area for i in props])

            # keep only largest item
            bw = morphology.remove_small_objects(labeled, min_size=size - 1)

            stack[i] = np.uint8(np.float32(bw) * intensity_threshold)

    return stack, nc_shape


def extract_video_frames(filepath, video_frames):
    # ignores blurring and intensity thresholding settings used in the process_data function above.

    data = xr.open_dataset(filepath)
    stack = np.asarray(data.images[video_frames])

    stack = stack[..., None]
    for t in tqdm(range(stack.shape[0])):
        for r in range(stack.shape[1]):
            for c in range(stack.shape[2]):
                stack[t, r, c] = cv2.cvtColor(stack[t, r, c], cv2.COLOR_BAYER_GB2RGB)[..., 1:2]

    print(stack.shape)
    return stack


def get_mcam_video_data(base_directory, filepaths, video_frames, superimpose_images, max_or_sum='max',
                        blur_sigma=None, intensity_threshold=None):
    # file_paths: list of nc files in the base_directory
    # video_frames: list of video frame indices
    # superipose_images: whether to do max projection across time

    if superimpose_images:
        if video_frames is None:
            stack_all = 0
            for i, filepath in tqdm(enumerate(filepaths), total=len(filepaths)):
                stack, nc_shape = process_data(base_directory + filepath, blur_sigma, intensity_threshold)
                if len(nc_shape) == 4:  # not a video
                    if max_or_sum == 'max':
                        stack_all = np.maximum(stack, stack_all)
                    elif max_or_sum == 'sum':
                        stack_all += np.float32(stack)
                elif len(nc_shape) == 5:  # video
                    # unflatten video dimension only:
                    stack = stack.reshape(nc_shape[0], nc_shape[1]*nc_shape[2], nc_shape[3], nc_shape[4], 1)
                    for i, frame in enumerate(stack):
                        if max_or_sum == 'max':
                            stack_all = np.maximum(frame, stack_all)
                        elif max_or_sum == 'sum':
                            stack_all += np.float32(stack)
        else:
            stack_all = 0
            for i, filepath in tqdm(enumerate(filepaths), total=len(filepaths)):
                stack_all_ = extract_video_frames(base_directory + filepath, video_frames)
                stack_all_ = stack_all_.max(0)
                stack_all_ = stack_all_.reshape((-1,) + stack_all_.shape[2:])
                if max_or_sum == 'max':
                    stack_all = np.maximum(stack_all, stack_all_)
                elif max_or_sum == 'sum':
                    stack_all += np.float32(stack_all_)
        stack = stack_all
        stack_list = [stack]
    else:
        if video_frames is None:
            stack_list = list()
            for filepath in tqdm(filepaths):
                stack, nc_shape = process_data(base_directory + filepath, blur_sigma)
                if len(nc_shape) == 4:  # not a video
                    pass
                elif len(nc_shape) == 5:  # video
                    # unflatten video dimension only:
                    # WARNING: INEFFICIENT!
                    stack = stack.reshape(nc_shape[0], nc_shape[1]*nc_shape[2], nc_shape[3], nc_shape[4], 1)
                    stack = stack[0]

                stack_list.append(stack)
        else:
            stack_list_list = list()
            for filepath in filepaths:
                stack = extract_video_frames(base_directory + filepath, video_frames)
                stack_list = stack.reshape((stack.shape[0], -1) + tuple(stack.shape[3:]))
                # e.g., shape^: len(video_frames), 54, L, W, 1
                stack_list_list.append(stack_list)
            stack_list = np.concatenate(stack_list_list, 0)

        stack = stack_list[0]  # for the para_mcam initialization

    return stack, stack_list


def summarize_recon(R, cmap='gray_r', colorbar=False):
    # for summarizing a 3D reconstruction: x, y, and z slices and max projections
    recon_size_x = R.shape[0]
    recon_size_y = R.shape[1]
    recon_size_z = R.shape[2]

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    R_ = np.copy(R).max(0)
    plt.imshow(R_, cmap=cmap)
    clim = np.percentile(R_, [.1, 99.9])
    plt.clim(clim)
    plt.title('projection across x')
    if colorbar:
        plt.colorbar()

    plt.subplot(2, 3, 4)
    R_ = R[recon_size_x // 2]
    plt.imshow(R_, cmap=cmap)
    clim = np.percentile(R_, [.1, 99.9])
    plt.clim(clim)
    plt.title('yz slice')
    if colorbar:
        plt.colorbar()

    plt.subplot(2, 3, 2)
    R_ = np.copy(R).T.max(1)
    plt.imshow(R_, cmap=cmap)
    clim = np.percentile(R_, [.1, 99.9])
    plt.clim(clim)
    plt.title('projection across y')
    if colorbar:
        plt.colorbar()

    plt.subplot(2, 3, 5)
    R_ = R[:, recon_size_y // 2, :].T
    plt.imshow(R_, cmap=cmap)
    clim = np.percentile(R_, [.1, 99.9])
    plt.clim(clim)
    plt.title('xz slice')
    if colorbar:
        plt.colorbar()

    plt.subplot(2, 3, 3)
    R_ = R.max(2)
    plt.imshow(R_, cmap=cmap)
    clim = np.percentile(R_, [.1, 99.9])
    plt.clim(clim)
    plt.title('en face projection')
    if colorbar:
        plt.colorbar()

    plt.subplot(2, 3, 6)
    R_ = R[:, :, recon_size_z // 2]
    plt.imshow(R_, cmap=cmap)
    clim = np.percentile(R_, [.1, 99.9])
    plt.clim(clim)
    plt.title('xy slice')
    if colorbar:
        plt.colorbar()
    plt.show()

