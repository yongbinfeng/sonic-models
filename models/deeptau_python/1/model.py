import numpy as np
import sys
import json

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        print("initialize model in the server")
        self.model_config = model_config = json.loads(args['model_config'])

        # Get output_tau configuration
        output_tau_config = pb_utils.get_output_config_by_name(
            model_config, "output_tau")

        # Get output_inner configuration
        output_inner_config = pb_utils.get_output_config_by_name(
            model_config, "output_inner")

        # Get output_outer configuration
        output_outer_config = pb_utils.get_output_config_by_name(
            model_config, "output_outer")

        # Convert Triton types to numpy types
        self.output_tau_dtype = pb_utils.triton_string_to_numpy(
            output_tau_config['data_type'])
        self.output_inner_dtype = pb_utils.triton_string_to_numpy(
            output_inner_config['data_type'])
        self.output_outer_dtype = pb_utils.triton_string_to_numpy(
            output_outer_config['data_type'])

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        output_tau_dtype = self.output_tau_dtype
        output_inner_dtype = self.output_inner_dtype
        output_outer_dtype = self.output_outer_dtype

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get INPUTs
            in_tau = pb_utils.get_input_tensor_by_name(request, "input_tau")
            in_inner_forconv = pb_utils.get_input_tensor_by_name(request, "input_inner_forconv")
            in_outer_forconv = pb_utils.get_input_tensor_by_name(request, "input_outer_forconv")
            in_innergrid_pos = pb_utils.get_input_tensor_by_name(request, "input_inner_pos")
            in_outergrid_pos = pb_utils.get_input_tensor_by_name(request, "input_outer_pos")

            in_tau = in_tau.as_numpy()

            # inner network
            in_inner_forconv = in_inner_forconv.as_numpy()
            # squeeze the two extra dimensions
            in_inner_forconv = np.squeeze(in_inner_forconv, axis=2)
            in_inner_forconv = np.squeeze(in_inner_forconv, axis=1)
            # take out the last row
            # as this should be the output when the input is zero
            zeropad_inner = in_inner_forconv[-1]
            in_inner_forconv = in_inner_forconv[:-1]

            # outer network
            in_outer_forconv = in_outer_forconv.as_numpy()
            in_outer_forconv = np.squeeze(in_outer_forconv, axis=2)
            in_outer_forconv = np.squeeze(in_outer_forconv, axis=1)
            # take out the last row
            # as this should be the output when the input is zero
            zeropad_outer = in_outer_forconv[-1]
            in_outer_forconv = in_outer_forconv[:-1]

            # grid indices information
            in_innergrid_pos = in_innergrid_pos.as_numpy()
            in_outergrid_pos = in_outergrid_pos.as_numpy()

            ntaus = in_tau.shape[0]
            output_inner_forconv = np.tile(zeropad_inner[np.newaxis, np.newaxis, np.newaxis, :], (ntaus, 11, 11, 1))
            output_outer_forconv = np.tile(zeropad_outer[np.newaxis, np.newaxis, np.newaxis, :], (ntaus, 21, 21, 1))

            # set the non-zero grids values to the outputs of the inner and outer network
            output_inner_forconv[tuple(in_innergrid_pos.T)] = in_inner_forconv
            output_outer_forconv[tuple(in_outergrid_pos.T)] = in_outer_forconv

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            out_tensor_tau   = pb_utils.Tensor("output_tau",
                                           in_tau.astype(output_tau_dtype))
            out_tensor_inner = pb_utils.Tensor("output_inner",
                                           output_inner_forconv.astype(output_inner_dtype))
            out_tensor_outer = pb_utils.Tensor("output_outer",
                                           output_outer_forconv.astype(output_outer_dtype))

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_tau, out_tensor_inner, out_tensor_outer])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
