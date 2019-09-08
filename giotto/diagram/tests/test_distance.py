import numpy as np
import sklearn.utils.testing as skt
import pytest
from giotto.diagram import distance

class Test_DiagramDistance:

# Testing parameters
    # Test for the wrong metric value
    def test_metric_V(self):
        X ={
            0: np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]]),
            2: np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]]),
            3: np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]])
            }
        dd = distance.DiagramDistance('bottleeck',
                                      metric_params={'n_samples': 200,
                                                     'delta': 0.0},
                                      n_jobs=None)
        skt.assert_raise_message(ValueError, "metric parameter has the "
                                 "wrong value: {}. "
                                 "Available values are: "
                                 "'bottleneck', 'wasserstein', 'landscape', "
                                 "'betti'.".format(dd.metric), dd.fit, X)

    # Test for the wrong n_sample type
    def test_n_samples_T(self):
        X ={
            0: np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]]),
            2: np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]]),
            3: np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]])
            }
        dd = distance.DiagramDistance('bottleneck',
                                      metric_params={'n_samples': 'a',
                                                     'delta': 0.0},
                                      n_jobs=None)
        skt.assert_raise_message(TypeError, "n_samples has the wrong type: %s."
                            " n_sample must be an integer "
                            "greater than "
                            "0." % type(dd.metric_params['n_samples']), dd.fit, X)

    # Test for the wrong n_sample value
    def test_n_samples_V(self):
        X ={
            0: np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]]),
            2: np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]]),
            3: np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]])
            }
        dd = distance.DiagramDistance('bottleneck',
                                      metric_params={'n_samples': -2,
                                                     'delta': 0.0},
                                      n_jobs=None)
        skt.assert_raise_message(ValueError, "n_samples has the "
                                 "wrong value: {}. n_sample must be an "
                                 "integer greater than"
                                 " 0.".format(dd.metric_params['n_samples']),
                                 dd.fit, X)

    # Test for the wrong delta value
    def test_delta_V(self):
        X ={
            0: np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]]),
            2: np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]]),
            3: np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]])
            }
        dd = distance.DiagramDistance('bottleneck',
                                      metric_params={'n_samples': 200,
                                                     'delta': -1},
                                      n_jobs=None)
        skt.assert_raise_message(ValueError, "delta has the wrong value: {}."
                             "delta must be a non-negative "
                             "integer.".format(dd.metric_params['delta']),
                             dd.fit, X)

    # Test for the wrong delta value
    def test_delta_T(self):
        X ={
            0: np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]]),
            2: np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]]),
            3: np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]])
            }
        dd = distance.DiagramDistance('bottleneck',
                                      metric_params={'n_samples': 200,
                                                     'delta': 'a'},
                                      n_jobs=None)
        skt.assert_raise_message(TypeError, "delta has the wrong type: %s."
                            "delta must be a non-negative " 
                            "integer." % type(dd.metric_params['delta']),
                            dd.fit, X)

    # Test for the wrong n_jobs value
    def test_n_jobs_V(self):
        X ={
            0: np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]]),
            2: np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]]),
            3: np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]])
            }
        dd = distance.DiagramDistance('bottleneck',
                                      metric_params={'n_samples': 200,
                                                     'delta': 0.0},
                                      n_jobs=-3)
        skt.assert_raise_message(ValueError, "n_jobs has the wrong value: {}."
                             " n_jobs must be equal to 'None' "
                             "or -1, or it must be an integer greater "
                             "than 0".format(dd.n_jobs), dd.fit, X)

# Testing inputs
    # Test for the wrong array key value
    def test_inputs_keys_V(self):
        X ={
            0: np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]]),
            -1: np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]]), # Wrong array key
            3: np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]])
            }
        dd = distance.DiagramDistance('bottleneck',
                                      metric_params={'n_samples': 200,
                                                     'delta': 0.0},
                                      n_jobs=None)
        skt.assert_raise_message(ValueError, "X keys must be non-negative"
                                 " integers.", dd.fit, X)

    # Test for the wrong array key type
    def test_inputs_keys_T(self):
        X ={
            0: np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]]),
            'a': np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]]), # Wrong array key
            3: np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]])
            }
        dd = distance.DiagramDistance('bottleneck',
                                      metric_params={'n_samples': 200,
                                                     'delta': 0.0},
                                      n_jobs=None)
        skt.assert_raise_message(TypeError, "X keys must be non-negative"
                                " integers.", dd.fit, X)

    # Test for the wrong structure dimension
    def test_inputs_arrayStruc_V(self):
        X ={
            0: np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]]),
            1: np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]]),
            3: np.array([[[1,1],[2,2]],[[4,4],[5,5],[6,6]]]) # Wrong array structure dimension
            }
        dd = distance.DiagramDistance('bottleneck',
                                      metric_params={'n_samples': 200,
                                                     'delta': 0.0},
                                      n_jobs=None)
        skt.assert_raise_message(ValueError, "Diagram structure dimension "
                                 "must be equal to 3.", dd.fit, X)

    # Test for the wrong 1st array dimension
    def test_inputs_arraydim1_V(self):
        X ={
            0: np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]]),
            1: np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]]),
            3: np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]],
                         [[4,4],[5,5],[6,6]]]) # Wrong array 1st dimension
            }
        dd = distance.DiagramDistance('bottleneck',
                                      metric_params={'n_samples': 200,
                                                     'delta': 0.0},
                                      n_jobs=None)
        skt.assert_raise_message(ValueError, "Diagram first dimension must "
                                 "be equal for all subarrays.", dd.fit, X)

    # Test for the wrong 3rd array dimension
    def test_inputs_arraydim3_V(self):
        X ={
            0: np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]]),
            1: np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]]),
            3: np.array([[[1],[2],[3]],[[4],[5],[6]]]) # Wrong array 3rd dimension
            }
        dd = distance.DiagramDistance('bottleneck',
                                      metric_params={'n_samples': 200,
                                                     'delta': 0.0},
                                      n_jobs=None)
        skt.assert_raise_message(ValueError, " Diagram coordinates dimension "
                                 "must be equal to 2.", dd.fit, X)

    # Test for the wrong value of a 3rd dimension array's elements
    def test_inputs_dim3_coord_V(self):
        X ={
            0: np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]]),
            1: np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]]),
            3: np.array([[[1,1],[2,2],[3,-1]],[[4,4],[5,5],[6,6]]]) # Wrong array element value
            }
        dd = distance.DiagramDistance('bottleneck',
                                      metric_params={'n_samples': 200,
                                                     'delta': 0.0},
                                      n_jobs=None)
        skt.assert_raise_message(ValueError, "Coordinates must be "
                                 "non-negative integers "
                                 "and the 2nd must be greater than or equal "
                                 "to the 1st one.", dd.fit, X)

    # Test for the wrong type of the 3rd array dimension
    def test_inputs_dim3_coord_T(self):
        X ={
            0: np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]]),
            1: np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]]),
            3: np.array([[[1,1],[2,2],[3,'a']],[[4,4],[5,5],[6,6]]]) # Wrong dimension type
            }
        dd = distance.DiagramDistance('bottleneck',
                                      metric_params={'n_samples': 200,
                                                     'delta': 0.0},
                                      n_jobs=None)
        skt.assert_raise_message(TypeError, "Coordinates must be "
                                 "non-negative integers "
                                 "and the 2nd must be greater than or equal "
                                 "to the 1st one.", dd.fit, X)

# Test no exception is raised with correct values of parameters and inputs
    def test_DiagramDistance_ENR(self):
        try:
            X ={
                0: np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]]),
                2: np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]]),
                3: np.array([[[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6]]])
                }
            dd = distance.DiagramDistance('bottleneck',
                                          metric_params={'n_samples': 200,
                                                         'delta': 0.0},
                                          n_jobs=None)
            dd.fit(X, y=None)
        except:
            print("Exception not expected. The inputs have the correct "
                  "structure.")
            raise ValueError
