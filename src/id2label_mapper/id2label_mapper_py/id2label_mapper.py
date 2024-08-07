#! /usr/bin/env python3
import rclpy
from rclpy.node import Node
from vision_msgs.msg import VisionClass, LabelInfo, VisionInfo
from ament_index_python.packages import get_package_share_directory

from id2label_mapper_services.srv import RegisterDatasetMapJSON, GetLocaltoGlobalMap, GetID2Label, GetDatasetID2Label

import json
import numpy as np
from array import array
from dataclasses import dataclass, field
from typing import Type, List, Dict, Optional

# Create a namespace
class Id2Label:
    # Define a type alias to indicate the label index length
    MsgType: Type[np.generic] = np.uint16
    MsgLargestVal = 2**16-1

@dataclass
class LabelEntry:
    label: str = "N/A"
    unique_id: Id2Label.MsgType = Id2Label.MsgLargestVal
    #   local_datasets[i]->local_labels[i]
    local_datasets: List[str] = field(default_factory=list)
    local_labels: List[Id2Label.MsgType] = field(default_factory=list)

class ClassIDMapper(Node):
    def __init__(self):
        """
        Initialize the ClassIDMapper node, services, and internal data structures.
        """
        super().__init__('id2label_mapper')
        # Class data:
        self.data : List[LabelEntry] = []
        self.local_to_global_maps : Dict[str, Dict[Id2Label.MsgType, int]] = {}
        # self.id2label : Dict[str, Dict[Id2Label.MsgType, str]] = {}
        self.id2label = {}
        # This is Redundant, as LabelEntry contains local_datasets[i]->local_labels[i] mapping:
        self.local_id2label : Dict[str, Dict[Id2Label.MsgType, str]] = {}

        self.unique_id_counter = 0
        self.database_version = 0

        # Append the "no class" label
        self.data.append(LabelEntry())
        self.data_keys = [e.label for e in self.data]

        self.register_service = self.create_service(RegisterDatasetMapJSON, 'register_dataset_map_json', self.register_dataset_callback)
        self.get_local_to_global_service = self.create_service(GetLocaltoGlobalMap, 'get_local_to_global_map', self.get_local_to_global_map_callback)
        self.get_id_to_label_service = self.create_service(GetID2Label, 'get_id_to_label', self.get_id_to_label_callback)
        self.get_dataset_id_to_label_service = self.create_service(GetDatasetID2Label, 'get_dataset_id_to_label', self.get_dataset_id_to_label_callback)

        self.declare_parameter('database_version', 0)
        self.set_database_version()

    def find_matching_label(self, label: str) -> Optional[int]:
        for i, entry in enumerate(self.data):
            if entry.label == label:
                return i
        return None

    def add_dataset(self, dataset_name : str, dataset_id2label : Dict[Id2Label.MsgType, str]):
        local2global_map : Dict[Id2Label.MsgType, int] = {}
        for id, label in dataset_id2label.items():
            matching_entry_idx = self.find_matching_label(label)
            if matching_entry_idx is not None:
                # Same class label found in existing entry
                self.get_logger().debug(f"Same class '{label}' found! Map: {id} -> {self.data[matching_entry_idx].unique_id}.  Overlapping datasets:  {self.data[matching_entry_idx].local_datasets}")
                if dataset_name not in self.data[matching_entry_idx].local_datasets:
                    self.data[matching_entry_idx].local_datasets.append(dataset_name)
                    self.data[matching_entry_idx].local_labels.append(id)

                # Match lookup to existing id
                local2global_map[id] = self.data[matching_entry_idx].unique_id
            else:
                # Create New entry
                if (self.unique_id_counter == (2**16-1)):
                    self.get_logger.error("Database full.  Unable to add additional classes.")
                else:
                    entry = LabelEntry(label=label, 
                                    unique_id=self.unique_id_counter, 
                                    local_datasets=[dataset_name], 
                                    local_labels=[id],
                                    )
                    self.unique_id_counter += 1

                    self.data.append(entry)
                    local2global_map[id] = entry.unique_id

        self.local_to_global_maps[dataset_name] = local2global_map
        self.local_id2label[dataset_name] = dataset_id2label
        self.data_keys = [e.label for e in self.data]

        self.id2label = {e.unique_id: e.label for e in self.data}
        self.get_logger().info(f'Updated id2label:  {self.id2label}')
        return None

    def register_dataset_callback(self, request: RegisterDatasetMapJSON.Request, response: RegisterDatasetMapJSON.Response) -> RegisterDatasetMapJSON.Response:
        """
        Register a dataset using a JSON file.

        :param request:
                string json_id2label_filename
                string dataset_name
        :param response: 
                int32 success
                string error_msgs
        :return: response
        """
        if request.dataset_name in self.local_id2label.keys():
            response.success = 0
            return response

        try:
            dataset_path = get_package_share_directory('id2label_mapper') + '/' + request.json_id2label_filename
            with open(dataset_path, 'r') as file:
                data = json.load(file)
                err = self.add_dataset(request.dataset_name, data['id2label'])
                if err:
                    response.success = 1
                    response.error_msgs = err
                else:
                    response.success = 0
                    response.error_msgs = ""
            
                # Still update the database version incase a partial update of the dataset was possible
                self.database_version += 1
                self.set_database_version()
        except FileNotFoundError or IOError:
            response.success = 2
            response.error_msgs = str(e)
        return response

    def set_database_version(self):
        database_param = rclpy.parameter.Parameter(
            'database_version',
            rclpy.Parameter.Type.INTEGER,
            self.database_version,
        )
        self.set_parameters([database_param])

    def get_local_to_global_map_callback(self, request: GetLocaltoGlobalMap.Request, response: GetLocaltoGlobalMap.Response) -> GetLocaltoGlobalMap.Response:
        """
        Get the mapping from local dataset IDs to global unique IDs.

        :param request:
                string dataset_name
        :param response: 
                uint16[] dataset_ids
                uint16[] unique_ids
                int32 database_version
        :return: response
        """
        dataset_name = request.dataset_name
        if dataset_name not in self.local_to_global_maps.keys():
            # failure
            response.dataset_ids = []
            response.unique_ids = []
            response.database_version = -1
            return response

        response.dataset_ids = list(map(int, self.local_to_global_maps[dataset_name].keys()))
        response.unique_ids = list(map(int, self.local_to_global_maps[dataset_name].values()))
        response.database_version = self.database_version
        return response


    def get_id_to_label_callback(self, request: GetID2Label.Request, response: GetID2Label.Response) -> GetID2Label.Response:
        """
        Get the mapping from global unique IDs to labels.

        :param request:
                (No input parameters)
        :param response: 
                vision_msgs/VisionClass[] class_map
                int32 database_version
        :return: response
        """
        response.class_map = [VisionClass(class_id=e.unique_id, class_name=e.label) for e in self.data]
        response.database_version = self.database_version
        return response

    def get_dataset_id_to_label_callback(self, request: GetDatasetID2Label.Request, response: GetDatasetID2Label.Response) -> GetDatasetID2Label.Response:
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
        if request.dataset_name not in self.local_id2label.keys():
            response.class_map = LabelInfo()
            response.pipeline_info = VisionInfo()
            return response

        local_data = self.local_id2label[request.dataset_name]
        response.class_map = LabelInfo()
        response.class_map.class_map = [VisionClass(class_id=int(k), class_name=v) for k, v in local_data.items()]
        response.database_version = self.database_version
        response.pipeline_info = VisionInfo()
        return response

def main(args=None):
    """
    Main function to initialize the rclpy library and start the ClassIDMapper node.
    """
    rclpy.init(args=args)
    node = ClassIDMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
