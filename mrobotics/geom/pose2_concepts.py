from . import basic_ops
import numpy as np

class pose2:
    def __init__(self, x, y , theta):
        """
        optionally with 3x3 covariance matrix!
        """
        self.x = x
        self.y = y
        self.theta = basic_ops.wrap_angle(theta) # let's play safe! Make the representation unique!

    def get_xy(self):
        return np.array([self.x,self.y])
    def get_SO2(self):
        return basic_ops.rot_mat_2d(self.theta)

    def get_pose_as_np(self):
        return np.array([self.x, self.y, self.theta])

    def get_SE2(self):
        out = np.eye(3)
        out[:-1,:-1] = self.get_SO2()
        out[0,-1],  out[1,-1] = self.x, self.y
        return out
    
    def map_XY(self, xy_wrt_me):
        """ coordinate mapping of 2D points from observer to the reference frame

        where this pose2 object is interpreted as the pose of the observer wrt the reference frame
        """
        return basic_ops.apply_SE2_to_pts(xy_wrt_me,self.theta, self.x, self.y)

    def make_upgrade(self, cov: np.array):
        return pose2_wcov(self.x, self.y, self.theta, cov=cov)

    def inverse(self):
        xy = -(self.get_SO2().T)@self.get_xy()
        return pose2(*xy, -self.theta)
    
    def compose(self, pose2_right):
        x_new, y_new = map(float, self.get_SO2()@pose2_right.get_xy()+self.get_xy())
        theta_new = self.theta + pose2_right.theta
        return pose2(x_new, y_new, theta_new)
    
    def jac1_compose(self, pose2_right, pose2_new = None):
        """
        suppose we have a composition pose_new:  myself (+) pose2_right,
        jac1 is defined to be \partial pose_new / \partial myself

        by supplying (the correct) pose2_new, you can speed up the computation
        """
        if pose2_new is None:
            pose2_new = self.compose(pose2_right)
        jac1 = np.identity(3)
        jac1[0,2] = self.y - pose2_new.y
        jac1[1,2] = pose2_new.x - self.x
        return jac1
    def jac2_compose(self):
        """
        suppose we have a composition pose_new:  myself (+) pose2_right,
        jac2 is defined to be \partial pose_new / \partial pose2_right
        """
        jac2 = np.identity(3)
        jac2[:-1,:-1] = SO2(self.theta)
        return jac2

class pose2_wcov(pose2):
    def __init__(self, x, y , theta, cov: np.array):
            assert cov.shape == (3,3) and np.linalg.det(cov)>0, "expect a symmetric positive definite 3x3 np-array as the Covariance matrix!"
            super().__init__(x,y,theta)
            self.cov = cov
    def compose_with_cov(self, pose2_right):
        """the return pose2 also contains the composition's covariance as well

        the Jacobion is defined by first order partial derivative of the 
        composed_pose (3-vector) w.r.t the [myself^T pose2_right^T]^T

        assume pose2_right is uncorrelated with myself
        """
        pose2_new = self.compose(pose2_right)
        jac1 = self.jac1_compose(pose2_right, pose2_new)
        jac2 = self.jac2_compose()
        pose2_new_cov = jac1@self.cov@jac1.T + jac2@pose2_right.cov@jac2.T
        return pose2_wcov(*pose2_new.get_pose_as_np(), cov=pose2_new_cov)
