#! /usr/bin/env python3
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
# from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

import json

from id2label_mapper_services.srv import RegisterDatasetMapJSON, GetLocaltoGlobalMap, GetID2Label, GetDatasetID2Label

#
# Helper functions for service calls
#
def _call(node, cli, req):
    """
    Implemntation of:  
        return cli.call(req)

    From https://docs.ros.org/en/foxy/How-To-Guides/Using-callback-groups.html:
        | Future’s done-callback that needs to be executed during the execution of the function call, 
        |    but this callback is not directly visible to the user

    I don't understand..., so it is impossible to make this work using cli.call(req)?
       - I was only able to get it to work by using async and waiting... 
    """
    # response = cli.call(req) # Error:   See above
    future = cli.call_async(req)
    rclpy.spin_until_future_complete(node, future)
    response = future.result()
    return response

def register_dataset_map_json(node, file_name, dataset_name):
    """
    Get and run a RegisterDatasetMapJSON Service call
    """
    # node.get_logger().info(f'Registering dataset {file_name} under:   {dataset_name}')

    client_cb_group = None
    cli = node.create_client(RegisterDatasetMapJSON, 'register_dataset_map_json', callback_group=client_cb_group)
    while not cli.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('service not available, waiting again...')

    req = RegisterDatasetMapJSON.Request()
    req.json_id2label_filename = file_name
    req.dataset_name = dataset_name
    response = _call(node, cli, req)

    return response

def get_id2label(node):
    """
    Get and run a GetID2Label call
    """    
    # node.get_logger().info(f'Getting id2label')

    cli = node.create_client(GetID2Label, 'get_id_to_label')
    while not cli.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('service not available, waiting again...')

    req = GetID2Label.Request()
    response = _call(node, cli, req)
    return response

def get_dataset_id2label(node, dataset_name):
    """
    Get and run a GetDatasetID2Label call
    """
    # node.get_logger().info(f'Getting dataset id2label from:   {dataset_name}')

    cli = node.create_client(GetDatasetID2Label, 'get_dataset_id_to_label')
    while not cli.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('service not available, waiting again...')

    req = GetDatasetID2Label.Request()
    req.dataset_name = dataset_name
    response = _call(node, cli, req)
    return response

def get_local_to_global_map(node, dataset_name):
    """
    Get and run a GetDatasetID2Label call
    """
    # node.get_logger().info(f'Getting GetLocaltoGlobalMap')

    cli = node.create_client(GetLocaltoGlobalMap, 'get_local_to_global_map')
    while not cli.wait_for_service(timeout_sec=1.0):
        node.get_logger().info('service not available, waiting again...')

    req = GetLocaltoGlobalMap.Request()
    req.dataset_name = dataset_name
    response = _call(node, cli, req)
    return response

#
# Tests
#
def test_populate_database(node, dataset_name, file_name):
    """
    Populate the server with COCO2017 dataset mappings
    """
    # Load data onto server
    response = register_dataset_map_json(node, file_name, dataset_name)
    assert response.success == 0, f'Failed Registering dataset {file_name} under: {dataset_name} with error:  {response.error_msgs}'

    # Manually read this file data.  
    #   The dataset_path could also be used to create a new dataset mapping on the server
    dataset_path = get_package_share_directory('id2label_mapper') + '/' + file_name
    with open(dataset_path, 'r') as file:
        data = json.load(file)
    gt_id2label = data["id2label"]

    # Check that all labels are present on the server
    response = get_id2label(node)
    assert response.database_version > 0, f'Invalid dataset version {response.database_version} returned'
    id2label = {e.class_id: e.class_name for e in response.class_map}
    for label in gt_id2label.values():
        assert label in id2label.values(), f'Label {label} not found in \n{id2label}'
    
    # Verify the original labels are correctly stored on the server
    response = get_dataset_id2label(node, dataset_name)
    assert response.database_version > 0, f'Invalid dataset version {response.database_version} returned'
    dataset_id2label = {e.class_id: e.class_name for e in response.class_map.class_map} # note double index on msg
    for key in gt_id2label.keys():
        assert dataset_id2label[int(key)] == gt_id2label[str(key)], f'Key {key} does not match in id2label ({dataset_id2label[int(key)]} v.s. {gt_id2label[str(key)]})'

    # Verify the mapping
    response = get_local_to_global_map(node, dataset_name)
    assert response.database_version > 0, f'Invalid dataset version {response.database_version} returned'
    for dataset_id, unique_id in zip(response.dataset_ids, response.unique_ids):
        assert gt_id2label[str(dataset_id)] == id2label[unique_id], f'Missmatched mapping ({gt_id2label[str(dataset_id)]} v.s. {id2label[unique_id]})'


def main(args=None):
    """ """
    rclpy.init(args=args)
    node = rclpy.create_node('test_id2label_services')

    test_populate_database(node, 'COCO2017', 'coco2017_id2label.json')
    test_populate_database(node, 'ADE20K', 'ade20k_id2label.json')

    node.get_logger().info("All tests passed!")

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
