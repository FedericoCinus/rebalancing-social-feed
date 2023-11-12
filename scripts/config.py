import json
import numpy as np
'''
 task 0: undirected small graphs
 task 1: runtime with synthetic graphs
 task 2: modularity heatmaps with SBM
 task 3: directed large graphs
'''
def create_json_file():
    data = {'task_0': {'generation':
                           {'task_name': 'approximation',
                             'folder': "../datasets/experiments/",
                             'statistics': 1,
                             'numb_nodes': [100],
                             'models': [
                                       {"network_model": "erdos", 
                                         "network_param_name": "p", 
                                         "network_param_values": [.05],
                                        "opinion_models": ["uniform", "gaussian"]},
                                       {"network_model": "barabasi", 
                                          "network_param_name": "m", 
                                          "network_param_values": [4],
                                          "opinion_models": ["uniform", "gaussian"]},
                                       {"network_model": "cs_aarhus", 
                                         "network_param_name": "None", "network_param_values": [None],
                                         "opinion_models": ["uniform", "gaussian"]},
                                       {"network_model": "dimacs10-football", 
                                         "network_param_name": "None", "network_param_values": [None],
                                         "opinion_models": ["uniform", "gaussian"]},
                                       {"network_model": "moreno_beach_beach", 
                                         "network_param_name": "None", "network_param_values": [None],
                                         "opinion_models": ["uniform", "gaussian"]},
                                       {"network_model": "political-books", 
                                         "network_param_name": "None", "network_param_values": [None],
                                         "opinion_models": ["uniform", "gaussian"]},
                                       {"network_model": "students", 
                                         "network_param_name": "None", "network_param_values": [None],
                                         "opinion_models": ["uniform", "gaussian"]},
                                       ],
                             'polarizations': [5.,],
                            'budgets': [1.,]
                            },
                       'optimization':{'folder': "../datasets/experiments/",
                                       'methods': [{'name': 'GD_optimizer', 
                                                    'params': {'routine': 'ADAM', 
                                                              'grad_params': {'budget': 1.},
                                                              'lr_params': {'lr_routine': 'constant', 'lr_coeff': 0.02}
                                                              }
                                                   },
                                                   {'name': 'SCS', 
                                                   'params': {'eps': None, 'max_iters': None, 'verbosity': False}},

                                                  ]
                                     }
                      },
          
            
          'task_1': {'generation':
                            {'task_name': 'runtime',
                             'folder': "../datasets/experiments/",
                             'statistics': 10,
                             'numb_nodes': [10, 20, 30, 40, 50, 
                                            100, 200, 300, 400, 500,
                                            1000, 2000, 3000, 4000, 5000,
                                            10000, 20000, 30000, 40000, 50000][::-1],
                             'models': [{"network_model": "barabasi", 
                                          "network_param_name": "m", 
                                          "network_param_values": [2],
                                          "opinion_models": ["uniform"]}],
                             'polarizations': [5.,],
                             'budgets': [1.,]
                            },
                     'optimization':
                             {'folder': "../datasets/experiments/",
                              'methods': [{'name': 'GD_optimizer', 
                                          'params': {'routine': 'simple_GD', 
                                                     'grad_params': {"β1": .9, "β2": 0.999},
                                                     'lr_params': {'lr_routine': 'constant', 'lr_coeff': 'lip'},
                                                     'return_objectives': 0}},
                                         {'name': 'SCS', 'params': {'verbosity': False }}
                                         ]
                             }
                    },
            'task_2': {'generation':
                            {'task_name': 'heatmap',
                             'folder': "../datasets/experiments/",
                             'statistics': 10,
                             'numb_nodes': [1000,],
                             'models': [{"network_model": "sbm", 
                                          "network_param_name": "β", 
                                          "network_param_values": list(np.linspace(.1, 1., num=25)),
                                          "opinion_models": ["gaussian", "uniform"]}],
                             'polarizations': list(np.linspace(.1, 10., num=25)),
                             'budgets': [1.,]
                            },
                     'optimization':
                             {'folder': "../datasets/experiments/",
                              'methods': [{'name': 'opposite_view_heuristic', 'params': {'verbosity': False }},
                                             {'name': 'popularity_heuristic', 'params': {'verbosity': False }},
                                             {'name': 'neutral_view_heuristic', 'params': {'verbosity': False }},
                                          {'name': 'GD_optimizer', 
                                          'params': {'routine': 'simple_GD', 
                                                     'grad_params': {'budget': 1.},
                                                     'lr_params': {'lr_routine': 'constant', 'lr_coeff': .1}}},
                                         ]
                             }
                    },
            'task_3': {'generation':
                           {'task_name': 'approximation_large',
                             'folder': "../datasets/experiments/",
                             'statistics': 1,
                             'numb_nodes': [10000],
                             'models': [
                                        {"network_model": "barabasi", 
                                           "network_param_name": "m", 
                                           "network_param_values": [2, 8],
                                           "opinion_models": ["uniform", "gaussian"]},
                                        {"network_model": "erdos", 
                                           "network_param_name": "p", 
                                           "network_param_values": [16/10000, 2/10000],
                                           "opinion_models": ["uniform", "gaussian"]},
                                        {"network_model": "sbm", 
                                           "network_param_name": "β", 
                                           "network_param_values": [.1, .5],
                                           "opinion_models": ["uniform", "gaussian"]},
                                        {"network_model": "munmun_twitter_social", 
                                         "network_param_name": "None", "network_param_values": [None],
                                         "opinion_models": ["uniform", "gaussian"]},
                                        {"network_model": "soc-epinions1", 
                                         "network_param_name": "None", "network_param_values": [None],
                                         "opinion_models": ["uniform", "gaussian"]},
                                        {"network_model": "facebook-wosn-wall", 
                                             "network_param_name": "None", "network_param_values": [None],
                                             "opinion_models": ["uniform", "gaussian"]}, 
                                        {"network_model": "ego-twitter", 
                                             "network_param_name": "None", "network_param_values": [None],
                                             "opinion_models": ["uniform", "gaussian"]},
                                        {"network_model": "flickr-growth",
                                             "network_param_name": "None", "network_param_values": [None],
                                             "opinion_models": ["uniform", "gaussian"]},
                                        {"network_model": "digg-friends", 
                                             "network_param_name": "None", "network_param_values": [None],
                                             "opinion_models": ["uniform", "gaussian"]},
                                        {"network_model": "epinions_large", 
                                             "network_param_name": "None", "network_param_values": [None],
                                             "opinion_models": ["uniform", "gaussian"]},
                                        {"network_model": "brexit", 
                                             "network_param_name": "None", "network_param_values": [None],
                                             "opinion_models": ["brexit"]},
                                        {"network_model": "vaxNoVax", 
                                             "network_param_name": "None", "network_param_values": [None],
                                             "opinion_models": ["vaxNoVax"]},

                                       ],
                             'polarizations': [1, 5.,],
                             'budgets': [1.,]
                            },
                       'optimization':{'folder': "../datasets/experiments/",
                                       'methods': [{'name': 'opposite_view_heuristic', 'params': {'verbosity': False }},
                                                   {'name': 'popularity_heuristic', 'params': {'verbosity': False }},
                                                   {'name': 'neutral_view_heuristic', 'params': {'verbosity': False }},
                                                   {'name': 'GD_optimizer', 
                                                   'params': {'routine': 'simple_GD', 
                                                              'grad_params': {'budget': 1.},
                                                              'lr_params': {'lr_routine': 'constant', 'lr_coeff': 0.2}}
                                                   },
                                                   {'name': 'GD_optimizer', 
                                                   'params': {'routine': 'ADAM', 
                                                              'grad_params': {'budget': 1.},
                                                              'lr_params': {'lr_routine': 'constant', 'lr_coeff': 0.2}}
                                                   },
                                                   
                                                  ]
                                     }
                      },
            'task_4': {'generation':
                           {'task_name': 'approximation_curves',
                             'folder': "../datasets/experiments/",
                             'statistics': 1,
                             'numb_nodes': [10000],
                             'models': [{"network_model": "vaxNoVax", 
                                             "network_param_name": "None", "network_param_values": [None],
                                             "opinion_models": ["vaxNoVax"]},
                                        {"network_model": "brexit", 
                                             "network_param_name": "None", "network_param_values": [None],
                                             "opinion_models": ["brexit"]},
                                        {"network_model": "digg-friends", 
                                             "network_param_name": "None", "network_param_values": [None],
                                             "opinion_models": ["uniform", "gaussian"]},
                                        {"network_model": "epinions_large", 
                                             "network_param_name": "None", "network_param_values": [None],
                                             "opinion_models": ["uniform", "gaussian"]},
                                        {"network_model": "erdos", 
                                           "network_param_name": "p", 
                                           "network_param_values": [16/10000,],
                                           "opinion_models": ["uniform", "gaussian"]},
                                        

                                       ],
                             'polarizations': [5.,],
                             'budgets': [1.,]
                            },
                       'optimization':{'folder': "../datasets/experiments/",
                                       'methods': [{'name': 'opposite_view_heuristic', 'params': {'verbosity': False }},
                                                   {'name': 'popularity_heuristic', 'params': {'verbosity': False }},
                                                   {'name': 'neutral_view_heuristic', 'params': {'verbosity': False }},
                                                   {'name': 'GD_optimizer', 
                                                   'params': {'routine': 'ADAM', 
                                                              'grad_params': {'budget': 1.},
                                                              'lr_params': {'lr_routine': 'constant', 'lr_coeff': 0.2}}
                                                   },
                                                   {'name': 'GD_optimizer', 
                                                   'params': {'routine': 'ADAM', 
                                                              'grad_params': {'budget': .75},
                                                              'lr_params': {'lr_routine': 'constant', 'lr_coeff': 0.2}}
                                                   },
                                                   {'name': 'GD_optimizer', 
                                                   'params': {'routine': 'ADAM', 
                                                              'grad_params': {'budget': .5},
                                                              'lr_params': {'lr_routine': 'constant', 'lr_coeff': 0.2}}
                                                   },
                                                   {'name': 'GD_optimizer', 
                                                   'params': {'routine': 'ADAM', 
                                                              'grad_params': {'budget': .25},
                                                              'lr_params': {'lr_routine': 'constant', 'lr_coeff': 0.2}}
                                                   }
                                                  ]
                                     }
                      },          
        }
    json_string = json.dumps(data)
    with open('.json_parameters.json', 'w') as outfile:
        json.dump(json_string, outfile)
