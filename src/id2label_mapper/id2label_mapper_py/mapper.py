#! /usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D

import struct

from id2label_mapper_services.srv import RegisterDatasetMapJSON, GetLocaltoGlobalMap, GetID2Label, GetDatasetID2Label
from id2label_mapper_services.srv import MapImage

class Mapper(Node):
    def __init__(self):
        """
        This provides examples for mapping class ids from the local (dataset dependant) index to a unique, global id.
        
        Ideally, this would be implemented as a c++ component or use zero-copy static function.

        The implementation here as a service is for testing purposes.
        """
        super().__init__('id2label_mapper_srvs')
        self.dataset_mappings = {}
        
        # Create new service
        self.map_image_srv = self.create_service(MapImage, 'map_image', self.map_image_srv_callback)

        # Subscribe to services
        self.register_future_requests = {}
        self.local_to_global_future_requests = {}
        self.register_client = self.create_client(RegisterDatasetMapJSON, 'register_dataset_map_json')
        self.get_local_to_global_client = self.create_client(GetLocaltoGlobalMap, 'get_local_to_global_map')
        self.get_id_to_label_client = self.create_client(GetID2Label, 'get_id_to_label')
        self.get_dataset_id_to_label_client = self.create_client(GetDatasetID2Label, 'get_dataset_id_to_label')
        req_services = [self.register_client, 
                        self.get_local_to_global_client, 
                        self.get_id_to_label_client, 
                        self.get_dataset_id_to_label_client,
                        ]
        while not all([r.wait_for_service(timeout_sec=1.0) for r in req_services]):
            self.get_logger().info('Service not available, waiting...')

        self.declare_parameter('database_version', rclpy.Parameter.Type.INTEGER)
        self.dataset_version_tracker = 0

    def wait(self, futures):
        for future in futures:
            rclpy.spin_until_future_complete(self, future)

    def call_register_service(self, f, dataset_name):
        request = RegisterDatasetMapJSON.Request()
        request.json_id2label_filename = f
        request.dataset_name = dataset_name

        future = self.register_client.call_async(request)
        self.register_future_requests[future] = request

        future.add_done_callback(self.verify_register_service)

        return future

    def verify_register_service(self, future):
        try:
            response = future.result()
            request = self.register_future_requests[future]
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')
            return

        if response.success != 0:
            self.get_logger().error(f'Service call "RegisterDatasetMapJSON" failed because:    {response.error_msgs}')
        else:
            self.get_logger().info(f'RegisterDatasetMapJSON Response for {request.dataset_name}: success={response.success}, error_msgs={response.error_msgs}')

        del self.register_future_requests[future]

    def get_local_to_global_service(self, dataset_name):
        request = GetLocaltoGlobalMap.Request()
        request.dataset_name = dataset_name

        future = self.get_local_to_global_client.call_async(request)
        self.wait([future])
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')
            return None

        if response.database_version < 0:
            self.get_logger().error(f"Failed to get dataset mapping from:    {dataset_name}")
            return None

        if not self.check_dataset_version(response.database_version):
            self.get_logger().warn(f"Dataset version mismatch.  Current Response {response.database_version},   Server: {self.dataset_version_tracker}")

        print("Dataset:  ",  request.dataset_name)
        print("\tDataset Ids:  ",  response.dataset_ids)
        print("\tUnique Ids:  ",  response.unique_ids)
        self.dataset_mappings[request.dataset_name] = {loc: glob for loc, glob in zip(response.dataset_ids, response.unique_ids)}

        return response

    def check_dataset_version(self, version):
        """
        This function requires debugging
        """
        return True
        self.dataset_version_tracker = self.get_parameter('database_version').get_parameter_value().integer_value
        return version == self.dataset_version_tracker

    def _get_id2label(self):
        request = GetID2Label.Request()
        future = self.get_id_to_label_client.call_async(request)
        self.wait([future])
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')
            return None

        if not self.check_dataset_version(response.database_version):
            self.get_logger().warn(f"Dataset version mismatch.  Current Response {response.database_version},   Server: {self.dataset_version_tracker}")

        return response

    def get_id2label(self):
        request = self._get_id2label()

        id2label = {} 
        for i in range(0, len(request.class_map)):
            id2label[request.class_map[i].class_id] = request.class_map[i].class_name

        return id2label

    def _get_dataset_id2label(self, dataset_name):
        request = GetDatasetID2Label.Request()
        request.dataset_name = dataset_name

        future = self.get_dataset_id_to_label_client.call_async(request)
        self.wait([future])
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')
            return None

        if not self.check_dataset_version(response.database_version):
            self.get_logger().warn(f"Dataset version mismatch.  Current Response {response.database_version},   Server: {self.dataset_version_tracker}")

        return response

    def get_dataset_id2label(self, dataset_name='COCO2017'):
        """
        Get the mapping from dataset IDs to labels for a specific dataset.

        :param request:
                string dataset_name
        :param response: 
                vision_msgs/LabelInfo class_map
                vision_msgs/VisionInfo pipeline_info
                int32 database_version
        :return: response
        """
        request = self._get_dataset_id2label(dataset_name)

        id2label = {} 
        for i in range(0, len(request.class_map.class_map)):
            id2label[request.class_map.class_map[i].class_id] = request.class_map.class_map[i].class_name

        return id2label


    def map_image_srv_callback(self, request: MapImage.Request, response: MapImage.Response) -> MapImage.Response:
        """
        Map an image from local dataset indexing to a global index
        Image format should be:  "mono8" or "mono16"
        Sets failure in the case where the dataset has not been loaded

        :param request:
                sensor_msgs/Image seg_map
                string dataset_name
        :param response: 
                sensor_msgs/Image seg_map
                int32 success
                string error_msgs
        :return: response

        """
        self.get_logger().info('map_image_srv_callback...')
        # print(self.dataset_mappings)


        # Perform the mapping from local to global IDs
        new_image = self.convert_image(request.seg_map, self.dataset_mappings[request.dataset_name])
        response.success = 1
        response.error_msgs = ""
        response.seg_map = new_image
        return response
        

    def convert_image(self, image, mapping):
        print("Mapping:  ",  dict(sorted(mapping.items())))
        if image.encoding == "mono8":
            return self.convert_image_mono8(image, mapping)
        else:
            return self.convert_image_mono16(image, mapping)

    def convert_image_mono8(self, image, mapping):
        """
        Convert the image using the provided mapping from local to global IDs.

        :param image: sensor_msgs/Image
        :param mapping: dictionary mapping local IDs to global IDs
        :return: sensor_msgs/Image with converted IDs
        """
        # Example of image processing: assuming mono8 to mono16 conversion
        new_image = Image()
        new_image.header = image.header
        new_image.height = image.height
        new_image.width = image.width
        new_image.encoding = 'mono16'
        new_image.is_bigendian = image.is_bigendian
        new_image.step = image.width * 2  # 2 bytes per pixel for mono16

        # Assuming the input image is mono8
        new_image_data = bytearray(new_image.height * new_image.step)
        for i in range(len(image.data)):
            local_id = image.data[i]
            global_id = mapping.get(local_id, 0)  # Default to 0 if local_id is not in mapping

        new_image.data = new_image_data
        return new_image
    
    def convert_image_mono16(self, image, mapping):
        """
        Convert the image using the provided mapping from local to global IDs.

        :param image: sensor_msgs/Image
        :param mapping: dictionary mapping local IDs to global IDs
        :return: sensor_msgs/Image with converted IDs
        """
        # Example of image processing: assuming mono8 to mono16 conversion
        new_image = Image()
        new_image.header = image.header
        new_image.height = image.height
        new_image.width = image.width
        new_image.encoding = 'mono16'
        new_image.is_bigendian = image.is_bigendian
        byteorder = "big" if new_image.is_bigendian else "little"
        byteorderchar = ">" if new_image.is_bigendian else "<"
        new_image.step = image.step
        new_image.data = image.data

        # Modify the values in the data
        for i in range(int(len(image.data)/2)):
            # local_id = image.data[i]
            # local_id = struct.unpack(byteorderchar+"H", image.data[i])
            local_id = image.data[i]
            # print("Local id: ", local_id)
            global_id = mapping.get(local_id, 0)  # Default to 0 if local_id is not in mapping
            # print("Global id: ", global_id)
            uint16_bytes = global_id.to_bytes(2, byteorder=byteorder, signed=False)
            new_image.data[i*2] = uint16_bytes[0]
            new_image.data[i*2+1] = uint16_bytes[1]
            # new_image.data[2*i:2*i+2] = bytearray(global_id.to_bytes(2, byteorder=byteorder, signed=False))
        # print("DONEONE")
        return new_image


def test_map_image_srv_callback(id2label_mapper):
    from cv_bridge import CvBridge, CvBridgeError
    import numpy as np
    bridge = CvBridge()
    request = MapImage.Request()
    request.dataset_name = "COCO2017"
    seg_map = np.ones((5,5), dtype=np.uint16)
    request.seg_map = bridge.cv2_to_imgmsg(seg_map, encoding='mono16') 
    print(request.seg_map)
    response = MapImage.Response()
    response = id2label_mapper.map_image_srv_callback(request, response)
    conv_seg_map = bridge.imgmsg_to_cv2(response.seg_map, desired_encoding='mono16')

    print("Seg Map:  ",  seg_map)
    print("Labels Converted:  ",  conv_seg_map)



def main(args=None):
    """
    Main function to initialize the rclpy library and start the Mapper node.
    """
    rclpy.init(args=args)
    id2label_mapper = Mapper()

    def wait(futures):
        for future in futures:
            rclpy.spin_until_future_complete(id2label_mapper, future)

    # Add datasets to the library
    future1 = id2label_mapper.call_register_service(f='coco2017_id2label.json', dataset_name='COCO2017')        
    future2 = id2label_mapper.call_register_service(f='ade20k_id2label.json', dataset_name='ADE20K')
    wait([future1, future2])

    # Check the mappings
    dataset_name = "COCO2017"
    response = id2label_mapper.get_local_to_global_service(dataset_name=dataset_name)
    # id2label_mapper.get_logger().info(f'{dataset_name} local_to_global_map={[(dataset, unique) for dataset, unique in zip(response.dataset_ids, response.unique_ids)]}, database_version={response.database_version}')
    dataset_name = "ADE20K"
    response = id2label_mapper.get_local_to_global_service(dataset_name=dataset_name)
    # id2label_mapper.get_logger().info(f'{dataset_name} local_to_global_map={[(dataset, unique) for dataset, unique in zip(response.dataset_ids, response.unique_ids)]}, database_version={response.database_version}')


    # Get global id2label mapping 
    id2label = id2label_mapper.get_id2label()
    id2label_ade20k = id2label_mapper.get_dataset_id2label("ADE20K")
    id2label_coco2017 = id2label_mapper.get_dataset_id2label("COCO2017")
    # print(id2label)
     
    overlapping_items = 22 # ['bed', 'person', 'door', 'chair', ...]
    print(len(id2label))
    assert( len(id2label_ade20k) == 150 )
    assert( len(id2label_coco2017) == 91 )
    assert( len(id2label) == (241-overlapping_items) )


    test_map_image_srv_callback(id2label_mapper)
    # Image seg_map()

    try:
        rclpy.spin(id2label_mapper)
    except KeyboardInterrupt:
        pass
    id2label_mapper.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
