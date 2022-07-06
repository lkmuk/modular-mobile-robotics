import numpy as np
from mrobotics.geom import pose2

def test_inverse():
    test_pose = pose2(0.4,-0.3,0.55)
    test_pose_inv = test_pose.inverse()
    assert np.allclose(test_pose.compose(test_pose_inv).get_pose_as_np(),0.0, atol=1e-8)
    assert np.allclose(test_pose_inv.compose(test_pose).get_pose_as_np(),0.0, atol=1e-8)

def test_coordinate_mapping():
    pose_sensor = pose2(1.0, 2.0, np.pi/2)
    xy_sensor = np.array([
        [0.0, 0.0],
        [20.0, -15.0],
        [-20.0, 100.0],
    ])

    # --------------------------------
    # calculate the correct value
    # 1. rotation
    vectors_sensorOrigin_to_xy_in_ref_frame =  np.array([ 
        [0.0, 0.0],
        [15.0, 20.0],
        [-100.0, -20.0]
    ])
    # 2. origin offset (translation)
    XY_ref_correct = vectors_sensorOrigin_to_xy_in_ref_frame + np.array([1.0, 2.0]).reshape(-1,2)

    # --------------------------------
    # validate the verification
    # print("the correct result should be", XY_ref_correct) 

    # --------------------------------
    XY_ref_calc = pose_sensor.map_XY(xy_sensor)
    assert np.allclose(XY_ref_correct, XY_ref_calc, rtol=1e-8)

if __name__ == "__main__":
    test_inverse()
    test_coordinate_mapping()
