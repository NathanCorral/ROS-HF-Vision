#! /usr/bin/env python3
import rclpy
from rclpy.node import Node
from vision_msgs.msg import VisionClass, LabelInfo, VisionInfo
from ament_index_python.packages import get_package_share_directory

from id2label_mapper_services.srv import RegisterDatasetMapJSON, GetLocaltoGlobalMap, GetID2Label, GetDatasetID2Label

import json
import numpy as np
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
        # Data managed by the class:
        self.data : List[LabelEntry] = []
        # Track elements (redundantly), to allow quick returning from service calls
        self.local_to_global_maps : Dict[str, (np.array, np.array)] = {}
        self.id2label : List[VisionClass] = []
        self.dataset_id2label : Dict[str, List[VisionClass]] = {}

        # Database elements
        self.unique_id_counter = 0
        self.database_version = 0

        # Register Services
        self.register_service = self.create_service(RegisterDatasetMapJSON, 'register_dataset_map_json', self.register_dataset_callback)
        self.get_local_to_global_service = self.create_service(GetLocaltoGlobalMap, 'get_local_to_global_map', self.get_local_to_global_map_callback)
        self.get_id_to_label_service = self.create_service(GetID2Label, 'get_id_to_label', self.get_id_to_label_callback)
        self.get_dataset_id_to_label_service = self.create_service(GetDatasetID2Label, 'get_dataset_id_to_label', self.get_dataset_id_to_label_callback)

        # Create a global parameter
        self.declare_parameter('database_version', 0)
        self.set_database_version()

    def add_dataset(self, dataset_name : str, json_id2label):
        """
        Add a new dataset and map its local IDs to global unique IDs.

        Parameters:
        dataset_name (str): The name of the dataset to add.
        json_id2label: A dictionary mapping local IDs to labels for the dataset.

        Returns: None
        """
        local2global_map_dataset_ids = []
        local2global_map_unique_ids = []
        dataset_id2label_list = []

        for id, label in json_id2label.items():
            dataset_id2label_list.append(VisionClass(class_id=int(id), class_name=label))

            matching_entry_idx = self.find_matching_label(label)
            if matching_entry_idx is not None:
                # Same class label found in existing entry
                # self.get_logger().debug(f"Same class '{label}' found! Map: {id} -> {self.data[matching_entry_idx].unique_id}.  Overlapping datasets:  {self.data[matching_entry_idx].local_datasets}")
                if dataset_name not in self.data[matching_entry_idx].local_datasets:
                    self.data[matching_entry_idx].local_datasets.append(dataset_name)
                    self.data[matching_entry_idx].local_labels.append(id)

                # Match lookup to existing id
                local2global_map_dataset_ids.append(int(id))
                local2global_map_unique_ids.append(int(self.data[matching_entry_idx].unique_id))

            else:
                # Create New entry
                if (self.unique_id_counter == Id2Label.MsgLargestVal):
                    self.get_logger().error("Database full.  Unable to add additional classes.")
                    break

                entry = LabelEntry(label=label, 
                                unique_id=self.unique_id_counter, 
                                local_datasets=[dataset_name], 
                                local_labels=[id],
                                )
                self.data.append(entry)
                local2global_map_dataset_ids.append(int(id))
                local2global_map_unique_ids.append(int(entry.unique_id))
                self.id2label.append(VisionClass(class_id=int(entry.unique_id), class_name=label))
                # dataset_id2label.class_map.class_map.append(VisionClass(class_id=id, class_name=label))

                self.unique_id_counter += 1

        self.local_to_global_maps[dataset_name] = (local2global_map_dataset_ids,
                                                    local2global_map_unique_ids)
        self.dataset_id2label[dataset_name] = dataset_id2label_list

        self.database_version += 1
        self.set_database_version()
        return

    def set_database_version(self):
        """
        Update the parameter server for the database version.

        Returns: None
        """
        database_param = rclpy.parameter.Parameter(
            'database_version',
            rclpy.Parameter.Type.INTEGER,
            self.database_version,
        )
        self.set_parameters([database_param])
    
    def find_matching_label(self, label: str) -> Optional[int]:
        """
        Find the index of an existing label in the dataset.
        
        Parameters:
            label (str): The label to search for in the dataset.

        Returns:
            Optional[int]: The index of the matching label in the dataset, or None if not found.
        """
        for i, entry in enumerate(self.data):
            if entry.label == label:
                return i
        return None

    def register_dataset_callback(self, request: RegisterDatasetMapJSON.Request, response: RegisterDatasetMapJSON.Response) -> RegisterDatasetMapJSON.Response:
        """
        Callback function for registering a dataset using a JSON file.

        Parameters:
            request (RegisterDatasetMapJSON.Request): The request object containing the dataset name and JSON file path.
            response (RegisterDatasetMapJSON.Response): The response object to be populated with success or error information.

        Returns:
            RegisterDatasetMapJSON.Response: The response with the success flag and error message (if any).
        """
        # Check if dataset has already been registered
        if request.dataset_name in self.local_to_global_maps.keys():
            response.success = 0
            return response

        try:
            dataset_path = get_package_share_directory('id2label_mapper') + '/' + request.json_id2label_filename
            with open(dataset_path, 'r') as file:
                data = json.load(file)
                self.add_dataset(request.dataset_name, data['id2label'])
        except FileNotFoundError or IOError:
            response.success = 2
            response.error_msgs = f"File not found:  {dataset_path}"

        return response

    def get_local_to_global_map_callback(self, request: GetLocaltoGlobalMap.Request, response: GetLocaltoGlobalMap.Response) -> GetLocaltoGlobalMap.Response:
        """
        Callback function to get the mapping from local dataset IDs to global unique IDs.

        Parameters:
            request (GetLocaltoGlobalMap.Request): The request object containing the dataset name.
            response (GetLocaltoGlobalMap.Response): The response object to be populated with the mapping and database version.

        Returns:
            GetLocaltoGlobalMap.Response: The response with the dataset IDs, unique IDs, and database version.
        """
        dataset_name = request.dataset_name
        if dataset_name not in self.local_to_global_maps.keys():
            # failure
            response.dataset_ids = []
            response.unique_ids = []
            response.database_version = -1
            return response

        response.dataset_ids = self.local_to_global_maps[dataset_name][0]
        response.unique_ids = self.local_to_global_maps[dataset_name][1]
        response.database_version = self.database_version
        return response

    def get_id_to_label_callback(self, request: GetID2Label.Request, response: GetID2Label.Response) -> GetID2Label.Response:
        """
        Callback function to get the global ID-to-label mapping.

        Parameters:
            request (GetID2Label.Request): The request object (no input parameters required).
            response (GetID2Label.Response): The response object to be populated with the class map and database version.

        Returns:
            GetID2Label.Response: The response with the class map and database version.
        """
        response.class_map = self.id2label
        response.database_version = self.database_version
        return response

    def get_dataset_id_to_label_callback(self, request: GetDatasetID2Label.Request, response: GetDatasetID2Label.Response) -> GetDatasetID2Label.Response:
        """
        Callback function to get the dataset-specific ID-to-label mapping.

        Parameters:
            request (GetDatasetID2Label.Request): The request object containing the dataset name.
            response (GetDatasetID2Label.Response): The response object to be populated with the class map, pipeline info, and database version.

        Returns:
            GetDatasetID2Label.Response: The response with the class map, pipeline info, and database version.
        """
        if request.dataset_name not in self.dataset_id2label.keys():
            response.class_map = LabelInfo()
            response.pipeline_info = VisionInfo()
            response.database_version = -1
            return response

        response.class_map.class_map = self.dataset_id2label[request.dataset_name]
        response.database_version = self.database_version
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
