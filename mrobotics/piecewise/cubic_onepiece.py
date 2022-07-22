import numpy as np
import matplotlib.pyplot as plt

class spline_2d_onepiece:
    def __init__(self,t0, t1, coeff=None):
        r""" single-piece cubic spline in 2D

        x(t) = \sum_{i=0}^{3} (t-t0/ t1-t0)^{i} cx[i]
        y(t) = \sum_{i=0}^{3} (t-t0/ t1-t0)^{i} cy[i]

        We assume t1 != t0

        We internally adopt the normalized interval.
        so you can apply affine scaling (modifying t0 and/or t1) 
        w/o the need of "refitting" the coefficients.

        We store the coefficient as a 4 x 2 table:

        cx[0] cy[0]
        cx[1] cy[1]
        cx[2] cy[2]
        cx[3] cy[3]

        They are initialized to be empty so you should first
        "fit" the coefficients before using it.

        You can alternatively pass in the coefficient 
        (maybe useful for piecewise extension)
        """
        if coeff is None:
            self.coeff = np.zeros((4,2))
        else:
            assert self.coeff.shape == (4,2)
        
        self.t0 = float(t0)
        self.T = float(t1-t0)
        # save some computation / expose problem in case t1 "==" t0 early on.
        self.oneOverT = 1/self.T 
    
    def get_ord(self):
        """order = number of polynomials (including the constant term)
        """
        return 4
    def get_num_dim(self):
        return 2 
       
    def normalize_parameter(self, t_eval):
        if isinstance(t_eval, float):
            t = np.array([t_eval])
        else: # assumed to be np array
            assert t_eval.ndim == 1
            t = t_eval
        t_normalized = (t-self.t0)*self.oneOverT
        return t_normalized
         
    def fit_pos_vel_ends(self, p0, v0, p1, v1):
        """a specialized algorithm for evaluating the coefficients
        
        Determine the 4 polynomial coefficients (per spatial dimension)
        all of the inputs are array-like objects of length 2
        """      
        # v0, v1 are generally not normalized wrt the (0,1) domain !
        p0_np = np.array(p0).reshape(2,-1)
        v0_np = self.T*np.array(v0).reshape(2,-1) # watch out!
        p1_np = np.array(p1).reshape(2,-1)
        v1_np = self.T*np.array(v1).reshape(2,-1) # watch out!
        
        self.coeff[0,:] = p0_np.T
        self.coeff[1,:] = v0_np.T
              
        vecA, vecB = p1_np-p0_np-v0_np, v1_np-v0_np
        self.coeff[2,:] = (3*vecA-vecB).reshape(-1)  # c2
        self.coeff[3,:] = (-2*vecA+vecB).reshape(-1) # c3


    # ----------------------------------------
    #  with respect to the normalized domain
    # ----------------------------------------
    def get_pos_wrt_natural(self, t_eval_normalized):
        """
        return shape: len(t_eval_normalized) x 2
        """
        basis_eval = np.ones((len(t_eval_normalized),self.get_ord()))
        for i in range(1, self.get_ord()):
            basis_eval[:,i] = t_eval_normalized**i
        
        return basis_eval@self.coeff
    
    def get_tang_wrt_natural(self, t_eval_normalized):
        """
        return shape: len(t_eval_normalized) x 2
        """
        # in general, a quadratic function wrt t
        basis_eval = np.ones((len(t_eval_normalized),self.get_ord()-1))
        for i in range(1,basis_eval.shape[-1]):
            # i is the polynomial degree in the tangent vector expression
            # i == polynomial degree (in the zeroth-order derivative) - 1
            # the basis row vectors for i == 0 are always 1, 
            # the one-initialization strategy means there is no need for looping i = 0
            basis_eval[:,i] = (i+1)*(t_eval_normalized**(i))
        # print(basis_eval)
        return basis_eval@self.coeff[1:,:]

    def get_deri_tang_wrt_natural(self,t_eval_normalized):
        # in general, a linear function wrt t

        # basis_eval = np.zeros((len(t_eval_normalized),self.get_ord()-2))
        # for i in range(0,basis_eval.shape[-1]):
        #     basis_eval[:,i] = (i+2)*(i+1)*(t_eval_normalized**(i))

        # more explicit
        basis_eval = np.zeros((len(t_eval_normalized),2))
        basis_eval[:,0] = 2
        basis_eval[:,1] = 6*t_eval_normalized
        return basis_eval@self.coeff[2:,:]

    def get_tang_curvature_wrt_natural(self, t_eval_normalized):
        """
        use get_tang(t_eval) or the normalized version if you ONLY need the tangent!
            inputs
            -------------
            t_eval_normalized (1d np vector of size N)
    
            outputs
            -------------
            tangent_wrt_natural: 2 x N
            signed_curvature: 1 x N
                ("left turn" is defined to have +ve curvature)
    
        """           
        tangent_vecs = self.get_tang_wrt_natural(t_eval_normalized)
        
        ddot_vecs = self.get_deri_tang_wrt_natural(t_eval_normalized)
        # unsigned curvature
        #       \| \dot{p} \cross \ddot{p} \|
        # k(t)= ------------------------------
        #        \| \dot{p} \| ^3
        
        # 1. the numerator (specialized for the 2D case, which is already in the correct polarity)
        signed_curvature = tangent_vecs[:,0]*ddot_vecs[:,1] - tangent_vecs[:,1]*ddot_vecs[:,0]
        # 2. the denominator
        signed_curvature /= (np.linalg.norm(tangent_vecs,axis=1, ord=2)**3)
        
        return tangent_vecs, signed_curvature

    # ----------------------------------------
    #  with respect to the user-defined domain
    # ----------------------------------------
    def get_pos(self, t_eval):
        t_normalized = self.normalize_parameter(t_eval)
        return self.get_pos_wrt_natural(t_normalized)
    
    def get_tang(self, t_eval):
        t_normalized = self.normalize_parameter(t_eval)
        return self.get_tang_wrt_natural(t_normalized)*self.oneOverT
    
    def get_deri_tang(self, t_eval):
        t_normalized = self.normalize_parameter(t_eval)
        return self.get_deri_tang_wrt_natural(t_normalized)*(self.oneOverT**2)

    def get_tang_curvature(self, t_eval):
        t_normalized = self.normalize_parameter(t_eval)
        tangent_wrt_natural, curvature = self.get_tang_curvature_wrt_natural(t_normalized)
        return tangent_wrt_natural*self.oneOverT, curvature

    # ----------------------------------------
    # more utilities
    # ----------------------------------------
    def calc_arc_length(self, n_samples = 1000):      
        # Python-optimized computation (instead of in-place computation)
        
        # avoid unnecessary computation of in the t-domain
        t_eval_normalized = np.linspace(0,1, n_samples+1)[:-1]
        vel_wrt_natural = self.get_tang_wrt_natural(t_eval_normalized)
        stage_dist = np.linalg.norm(vel_wrt_natural,axis=1,ord=2) # *self.oneOverT # overdone it
        # finally, the Riemann sum 
        return np.sum(stage_dist)/n_samples # *(1.0-0.0)
    
    def vis_curve(self, ax=None, n_samples = 100, **plot_kwargs):
        """

        Args:
            ax (matplotlib pylab/ pyplot axis, optional)
            n_samples (int, optional): Defaults to 100.
        """
        if ax is None:
            _, ax = plt.subplots()
        t = np.linspace(self.t0, self.t0 + self.T, n_samples)
        xy = self.get_pos(t)
        ax.plot(*xy.T, **plot_kwargs)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')