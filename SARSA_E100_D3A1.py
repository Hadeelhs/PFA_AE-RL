#origial
ismModifDoss="Original_code_SARSA_E100_D3A1"
ismModif=""
ismModifDataset="original_sarsa_E100_D3A1"

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, clone_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import json
from sklearn.utils import shuffle
import os
import sys
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score, recall_score, precision_score
import pandas as pd
import tensorflow as tf
import random

# New imports for the added section
import itertools
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress INFO, WARNING, and DEBUG messages
tf.get_logger().setLevel('ERROR')  # Only show errors from TensorFlow's logger

# Set seeds for reproducibility
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)
tf.random.set_seed(random_seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
gpus = tf.config.list_physical_devices('GPU')

def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = abs_error - quadratic
    return tf.reduce_mean(0.5 * tf.square(quadratic) + delta * linear)
    
# Needed for keras huber_loss locate
import tensorflow.keras.losses
tensorflow.keras.losses.huber_loss = huber_loss

class data_cls:
    def __init__(self,train_test,**kwargs):
        col_names = ["duration","protocol_type","service","flag","src_bytes",
            "dst_bytes","land_f","wrong_fragment","urgent","hot","num_failed_logins",
            "logged_in","num_compromised","root_shell","su_attempted","num_root",
            "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
            "is_host_login","is_guest_login","count","srv_count","serror_rate",
            "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
            "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
            "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
            "dst_host_rerror_rate","dst_host_srv_rerror_rate","labels","dificulty"]
        self.index = 0
        # Data formated path and test path. 
        self.loaded = False
        self.train_test = train_test
        self.train_path = kwargs.get('train_path', '../datasets/NSL/KDDTrain+.txt')
        self.test_path = kwargs.get('test_path',
                                    'https://raw.githubusercontent.com/gcamfer/Anomaly-ReactionRL/master/datasets/NSL/KDDTest%2B.txt')
        
        self.formated_train_path = kwargs.get('formated_train_path', 
                                              "formated_train_adv_"+ismModifDataset+".data")
        self.formated_test_path = kwargs.get('formated_test_path',
                                             "formated_test_adv_"+ismModifDataset+".data")
        
        self.attack_types = ['normal','DoS','Probe','R2L','U2R']
        self.attack_names = []
        self.attack_map =   { 'normal': 'normal',
                        
                        'back': 'DoS',
                        'land': 'DoS',
                        'neptune': 'DoS',
                        'pod': 'DoS',
                        'smurf': 'DoS',
                        'teardrop': 'DoS',
                        'mailbomb': 'DoS',
                        'apache2': 'DoS',
                        'processtable': 'DoS',
                        'udpstorm': 'DoS',
                        
                        'ipsweep': 'Probe',
                        'nmap': 'Probe',
                        'portsweep': 'Probe',
                        'satan': 'Probe',
                        'mscan': 'Probe',
                        'saint': 'Probe',
                    
                        'ftp_write': 'R2L',
                        'guess_passwd': 'R2L',
                        'imap': 'R2L',
                        'multihop': 'R2L',
                        'phf': 'R2L',
                        'spy': 'R2L',
                        'warezclient': 'R2L',
                        'warezmaster': 'R2L',
                        'sendmail': 'R2L',
                        'named': 'R2L',
                        'snmpgetattack': 'R2L',
                        'snmpguess': 'R2L',
                        'xlock': 'R2L',
                        'xsnoop': 'R2L',
                        'worm': 'R2L',
                        
                        'buffer_overflow': 'U2R',
                        'loadmodule': 'U2R',
                        'perl': 'U2R',
                        'rootkit': 'U2R',
                        'httptunnel': 'U2R',
                        'ps': 'U2R',    
                        'sqlattack': 'U2R',
                        'xterm': 'U2R'
                    }
        self.all_attack_names = list(self.attack_map.keys())

        formated = False     
        
        # Test formated data exists
        if os.path.exists(self.formated_train_path) and os.path.exists(self.formated_test_path):
            formated = True
       
        self.formated_dir = "../dataset/formated/"
        if not os.path.exists(self.formated_dir):
            os.makedirs(self.formated_dir)
               
            
        # If it does not exist, it's needed to format the data
        if not formated:
            ''' Formating the dataset for ready-2-use data'''
            print("Start formating")
            self.df = pd.read_csv(self.train_path,sep=',',names=col_names,index_col=False)
            if 'dificulty' in self.df.columns:
                self.df.drop('dificulty', axis=1, inplace=True) #in case of difficulty     
                
            data2 = pd.read_csv(self.test_path,sep=',',names=col_names,index_col=False)
            if 'dificulty' in data2:
                del(data2['dificulty'])
            train_indx = self.df.shape[0]
            frames = [self.df,data2]
            self.df = pd.concat(frames)
            
            # Dataframe processing
            self.df = pd.concat([self.df.drop('protocol_type', axis=1), pd.get_dummies(self.df['protocol_type'])], axis=1)
            self.df = pd.concat([self.df.drop('service', axis=1), pd.get_dummies(self.df['service'])], axis=1)
            self.df = pd.concat([self.df.drop('flag', axis=1), pd.get_dummies(self.df['flag'])], axis=1)
              
            # 1 if ``su root'' command attempted; 0 otherwise 
            self.df['su_attempted'] = self.df['su_attempted'].replace(2.0, 0.0)
            
             # One hot encoding for labels
            self.df = pd.concat([self.df.drop('labels', axis=1),
                            pd.get_dummies(self.df['labels'])], axis=1)
            
            
            # Normalization of the df
            #normalized_df=(df-df.mean())/df.std()
            for indx,dtype in self.df.dtypes.iteritems():
                if dtype == 'float64' or dtype == 'int64':
                    if self.df[indx].max() == 0 and self.df[indx].min()== 0:
                        self.df[indx] = 0
                    else:
                        self.df[indx] = (self.df[indx]-self.df[indx].min())/(self.df[indx].max()-self.df[indx].min())
            
            
            # Save data
            test_df = self.df.iloc[train_indx:self.df.shape[0]]
            test_df = shuffle(test_df,random_state=np.random.randint(0,100))
            self.df = self.df[:train_indx]
            self.df = shuffle(self.df,random_state=np.random.randint(0,100))
            test_df.to_csv(self.formated_test_path,sep=',',index=False)
            self.df.to_csv(self.formated_train_path,sep=',',index=False)
            
            # Create a list with the existent attacks in the df
            for att in self.attack_map:
                if att in self.df.columns:
                # Add only if there is exist at least 1
                    if np.sum(self.df[att].values) > 1:
                        self.attack_names.append(att)

    def get_shape(self):
        if self.loaded is False:
            self._load_df()
        
        self.data_shape = self.df.shape
        # stata + labels
        return self.data_shape
    
    ''' Get n-rows from loaded data 
        The dataset must be loaded in RAM
    '''
    def get_batch(self,batch_size=100):
        if self.loaded is False:
            self._load_df()
        
        # Read the df rows
        indexes = list(range(self.index,self.index+batch_size))    
        if max(indexes)>self.data_shape[0]-1:
            dif = max(indexes)-self.data_shape[0]
            indexes[len(indexes)-dif-1:len(indexes)] = list(range(dif+1))
            self.index=batch_size-dif
            batch = self.df.iloc[indexes]
        else: 
            batch = self.df.iloc[indexes]
            self.index += batch_size    
            
        labels = batch[self.attack_names]
        
        batch = batch.drop(self.all_attack_names,axis=1)
            
        return batch,labels
    
    def get_full(self):
        if self.loaded is False:
            self._load_df()
            
        
        labels = self.df[self.attack_names]
        
        batch = self.df.drop(self.all_attack_names,axis=1)
        

        return batch,labels
      
    def _load_df(self):
        if self.train_test == 'train':
            self.df = pd.read_csv(self.formated_train_path,sep=',') # Read again the csv
        else:
            self.df = pd.read_csv(self.formated_test_path,sep=',')
        self.index=np.random.randint(0,self.df.shape[0]-1,dtype=np.int32)
        self.loaded = True
         # Create a list with the existent attacks in the df
        for att in self.attack_map:
            if att in self.df.columns:
                # Add only if there is exist at least 1
                if np.sum(self.df[att].values) > 1:
                    self.attack_names.append(att)
        #self.headers = list(self.df) 

from tensorflow.keras.models import clone_model

class QNetwork:
    def __init__(self, obs_size, num_actions, hidden_size=100, hidden_layers=1, learning_rate=.2):
        """Initialize Q-Network with TF 2.x compatible architecture"""
        # Create sequential model
        self.model = Sequential()
        
        # Add input layer
        self.model.add(Dense(
            hidden_size,
            activation='relu',
            input_dim=obs_size
        ))
        
        # Add hidden layers
        for _ in range(hidden_layers-1):
            self.model.add(Dense(
                hidden_size,
                activation='relu'
            ))
        
        # Add output layer
        self.model.add(Dense(num_actions))
        
        # Compile model
        self.model.compile(
            optimizer = optimizers.Adam(0.00025),
            loss=huber_loss
        )
    
    def predict(self, state, batch_size=None):
        """Predict Q-values for given state"""
        return self.model.predict(state, batch_size=batch_size)
    
    def train(self, states, q_values):
        """Train the network"""
        return self.model.train_on_batch(states, q_values)
    
    @staticmethod
    def copy_model(model):
        """Copy the model architecture and weights"""
        copied_model = clone_model(model)
        copied_model.set_weights(model.get_weights())
        return copied_model




#Policy interface
class Policy:
    def __init__(self, num_actions, estimator):
        self.num_actions = num_actions
        self.estimator = estimator
    
class Epsilon_greedy(Policy):
    def __init__(self,estimator ,num_actions ,epsilon,min_epsilon,decay_rate, epoch_length):
        Policy.__init__(self, num_actions, estimator)
        self.name = "Epsilon Greedy"
        
        if (epsilon is None or epsilon < 0 or epsilon > 1):
            print("EpsilonGreedy: Invalid value of epsilon", flush = True)
            sys.exit(0)
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.actions = list(range(num_actions))
        self.step_counter = 0
        self.epoch_length = epoch_length
        self.decay_rate = decay_rate
        
        #if epsilon is up 0.1, it will be decayed over time
        if self.epsilon > 0.01:
            self.epsilon_decay = True
        else:
            self.epsilon_decay = False
    
    def get_actions(self,states):
        # get next action
        if np.random.rand() <= self.epsilon:
            actions = np.random.randint(0, self.num_actions,states.shape[0])
        else:
            self.Q = self.estimator.predict(states,states.shape[0])
            actions = []
            for row in range(self.Q.shape[0]):
                best_actions = np.argwhere(self.Q[row] == np.amax(self.Q[row]))
                actions.append(best_actions[np.random.choice(len(best_actions))].item())
            
        self.step_counter += 1 
        # decay epsilon after each epoch
        if self.epsilon_decay:
            if self.step_counter % self.epoch_length == 0:
                self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate**self.step_counter)
            
        return actions
    
class ReplayMemory(object):
    """Implements basic replay memory"""

    def __init__(self, observation_size, max_size):
        self.observation_size = observation_size
        self.num_observed = 0
        self.max_size = max_size
        self.samples = {
                 'obs'      : np.zeros(self.max_size * 1 * self.observation_size,
                                       dtype=np.float32).reshape(self.max_size,self.observation_size),
                 'action'   : np.zeros(self.max_size * 1, dtype=np.int16).reshape(self.max_size, 1),
                 'reward'   : np.zeros(self.max_size * 1).reshape(self.max_size, 1),
                 'terminal' : np.zeros(self.max_size * 1, dtype=np.int16).reshape(self.max_size, 1),
               }

    def observe(self, state, action, reward, done):
        index = self.num_observed % self.max_size
        self.samples['obs'][index, :] = state
        self.samples['action'][index, :] = action
        self.samples['reward'][index, :] = reward
        self.samples['terminal'][index, :] = done

        self.num_observed += 1

    def sample_minibatch(self, minibatch_size):
        max_index = min(self.num_observed, self.max_size) - 1
        sampled_indices = np.random.randint(max_index, size=minibatch_size)

        s      = np.asarray(self.samples['obs'][sampled_indices, :], dtype=np.float32)
        s_next = np.asarray(self.samples['obs'][sampled_indices+1, :], dtype=np.float32)

        a      = self.samples['action'][sampled_indices].reshape(minibatch_size)
        r      = self.samples['reward'][sampled_indices].reshape((minibatch_size, 1))
        done   = self.samples['terminal'][sampled_indices].reshape((minibatch_size, 1))

        return (s, a, r, s_next, done)


'''
Reinforcement learning Agent definition
'''

class Agent(object):  
    def __init__(self, actions, obs_size, policy="EpsilonGreedy", **kwargs):
        self.actions = actions
        self.num_actions = len(actions)
        self.obs_size = obs_size
        
        self.epsilon = kwargs.get('epsilon', 1)
        self.min_epsilon = kwargs.get('min_epsilon', .1)
        self.gamma = kwargs.get('gamma', .001)
        self.minibatch_size = kwargs.get('minibatch_size', 2)
        self.epoch_length = kwargs.get('epoch_length', 100)
        self.decay_rate = kwargs.get('decay_rate',0.99)
        self.ExpRep = kwargs.get('ExpRep',True)
        if self.ExpRep:
            self.memory = ReplayMemory(self.obs_size, kwargs.get('mem_size', 10))
        
        self.ddqn_time = 100
        self.ddqn_update = self.ddqn_time

        self.model_network = QNetwork(self.obs_size, self.num_actions,
                                      kwargs.get('hidden_size', 100),
                                      kwargs.get('hidden_layers',1),
                                      kwargs.get('learning_rate',.2))
        self.target_model_network = QNetwork(self.obs_size, self.num_actions,
                                      kwargs.get('hidden_size', 100),
                                      kwargs.get('hidden_layers',1),
                                      kwargs.get('learning_rate',.2))
        self.target_model_network.model = QNetwork.copy_model(self.model_network.model)
        
        if policy == "EpsilonGreedy":
            self.policy = Epsilon_greedy(self.model_network, len(actions),
                                         self.epsilon, self.min_epsilon,
                                         self.decay_rate, self.epoch_length)
        
    def learn(self, states, actions, next_states, rewards, done):
        if self.ExpRep:
            self.memory.observe(states, actions, rewards, done)
        else:
            self.states = states
            self.actions = actions
            self.next_states = next_states
            self.rewards = rewards
            self.done = done
        
    def update_model(self):
        if self.ExpRep:
            (states, actions, rewards, next_states, done) = self.memory.sample_minibatch(self.minibatch_size)
        else:
            states = self.states
            rewards = self.rewards
            next_states = self.next_states
            actions = self.actions
            done = self.done
        
        # SARSA: Get next actions (a') using the current policy
        next_actions = self.policy.get_actions(next_states)
        
        # Compute Q-values for current and next states
        Q = self.model_network.predict(states, self.minibatch_size)
        Q_next = self.target_model_network.predict(next_states, self.minibatch_size)
        targets = Q.copy()
        
        # SARSA update: Use Q(s', a') instead of max Q(s', a')
        sx = np.arange(len(actions))
        targets[sx, actions] = rewards.reshape(Q[sx, actions].shape) + \
                               self.gamma * Q_next[sx, next_actions] * \
                               (1 - done.reshape(Q[sx, actions].shape))
        
        # Train the model
        loss = self.model_network.train(states, targets)
        
        # Update target network periodically (same as original DDQN)
        self.ddqn_update -= 1
        if self.ddqn_update == 0:
            self.ddqn_update = self.ddqn_time
            self.target_model_network.model.set_weights(self.model_network.model.get_weights())
        
        return loss
    
    def act(self, state, policy):
        raise NotImplementedError

class DefenderAgent(Agent):      
    def __init__(self, actions, obs_size, policy="EpsilonGreedy", **kwargs):
        super().__init__(actions,obs_size, policy="EpsilonGreedy", **kwargs)
        
    def act(self,states):
        # Get actions under the policy
        actions = self.policy.get_actions(states)
        return actions
    
class AttackAgent(Agent):      
    def __init__(self, actions, obs_size, policy="EpsilonGreedy", **kwargs):
        super().__init__(actions,obs_size, policy="EpsilonGreedy", **kwargs)
        
    def act(self,states):
        # Get actions under the policy
        actions = self.policy.get_actions(states)
        return actions


    
'''
Reinforcement learning Enviroment Definition
'''
class RLenv(data_cls):
    def __init__(self,train_test,**kwargs):
        data_cls.__init__(self,train_test)
        self.data_shape = data_cls.get_shape(self)
        self.batch_size = kwargs.get('batch_size',1) # experience replay -> batch = 1
        self.iterations_episode = kwargs.get('iterations_episode',10)
        if self.batch_size=='full':
            self.batch_size = int(self.data_shape[0]/iterations_episode)

    '''
    _update_state: function to update the current state
    Returns:
        None
    Modifies the self parameters involved in the state:
        self.state and self.labels
    Also modifies the true labels to get learning knowledge
    '''
    def _update_state(self):        
        self.states,self.labels = data_cls.get_batch(self)
        
        # Update statistics
        self.true_labels += np.sum(self.labels).values

    '''
    Returns:
        + Observation of the enviroment
    '''
    def reset(self):
        # Statistics
        self.def_true_labels = np.zeros(len(self.attack_types),dtype=int)
        self.def_estimated_labels = np.zeros(len(self.attack_types),dtype=int)
        self.att_true_labels = np.zeros(len(self.attack_names),dtype=int)
        
        self.state_numb = 0
        
        self.states,self.labels = data_cls.get_batch(self,self.batch_size)
        
        self.total_reward = 0
        self.steps_in_episode = 0
        return self.states.values 
   
    '''
    Returns:
        State: Next state for the game
        Reward: Actual reward
        done: If the game ends (no end in this case)
    
    In the adversarial enviroment, it's only needed to return the actual reward
    '''    
    def act(self,defender_actions,attack_actions):
        # Clear previous rewards        
        self.att_reward = np.zeros(len(attack_actions))       
        self.def_reward = np.zeros(len(defender_actions))
        
        
        attack = [self.attack_types.index(self.attack_map[self.attack_names[att]]) for att in attack_actions]
        
        self.def_reward = (np.asarray(defender_actions)==np.asarray(attack))*1
        self.att_reward = (np.asarray(defender_actions)!=np.asarray(attack))*1

         
       
        self.def_estimated_labels += np.bincount(defender_actions,minlength=len(self.attack_types))
        # TODO
        # list comprehension
        
        for act in attack_actions:
            self.def_true_labels[self.attack_types.index(self.attack_map[self.attack_names[act]])] += 1
        

        # Get new state and new true values 
        attack_actions = attacker_agent.act(self.states)
        self.states = env.get_states(attack_actions)
        
        # Done allways false in this continuous task       
        self.done = np.zeros(len(attack_actions),dtype=bool)
            
        return self.states, self.def_reward,self.att_reward, attack_actions, self.done
    
    '''
    Provide the actual states for the selected attacker actions
    Parameters:
        self:
        attacker_actions: optimum attacks selected by the attacker
            it can be one of attack_names list and select random of this
    Returns:
        State: Actual state for the selected attacks
    '''
    def get_states(self,attacker_actions):
        first = True
        for attack in attacker_actions:
            if first:
                minibatch = (self.df[self.df[self.attack_names[attack]]==1].sample(1))
                first = False
            else:
                minibatch=minibatch.append(self.df[self.df[self.attack_names[attack]]==1].sample(1))
        
        self.labels = minibatch[self.attack_names]
        minibatch.drop(self.all_attack_names,axis=1,inplace=True)
        self.states = minibatch
        
        return self.states

test_loss_chain = [] #MOD-1

if __name__ == "__main__":
    
    formated_test_path = "formated_test_adv_" + ismModifDataset + ".data" #MOD-1
    env_test = RLenv('test', formated_test_path=formated_test_path) #MOD-1
  
    kdd_train = "https://raw.githubusercontent.com/gcamfer/Anomaly-ReactionRL/master/datasets/NSL/KDDTrain%2B.txt"
    kdd_test = "https://raw.githubusercontent.com/gcamfer/Anomaly-ReactionRL/master/datasets/NSL/KDDTest%2B.txt"

    formated_train_path = "formated_train_adv_"+ismModifDataset+".data"
    formated_test_path = "formated_test_adv_"+ismModifDataset+".data"
    
    
    # Train batch
    batch_size = 1
    # batch of memory ExpRep
    minibatch_size = 100
    ExpRep = True
    
    iterations_episode = 100
  
    # Initialization of the enviroment
    env = RLenv('train',train_path=kdd_train,test_path=kdd_test,
                formated_train_path = formated_train_path,
                formated_test_path = formated_test_path,batch_size=batch_size,
                iterations_episode=iterations_episode)    
    # obs_size = size of the state
    obs_size = env.data_shape[1]-len(env.all_attack_names)
    
    #num_episodes = int(env.data_shape[0]/(iterations_episode)/10)
    num_episodes = 100 ############ kenet 100
    
    '''
    Definition for the defensor agent.
    '''
    defender_valid_actions = list(range(len(env.attack_types))) # only detect type of attack
    defender_num_actions = len(defender_valid_actions)    
    
	
    def_epsilon = 1 # exploration
    min_epsilon = 0.01 # min value for exploration
    def_gamma = 0.001
    def_decay_rate = 0.99
    
    def_hidden_size = 100
    def_hidden_layers = 3
    
    def_learning_rate = .2
    
    defender_agent = DefenderAgent(defender_valid_actions,obs_size,"EpsilonGreedy",
                          epoch_length = iterations_episode,
                          epsilon = def_epsilon,
                          min_epsilon = min_epsilon,
                          decay_rate = def_decay_rate,
                          gamma = def_gamma,
                          hidden_size=def_hidden_size,
                          hidden_layers=def_hidden_layers,
                          minibatch_size = minibatch_size,
                          mem_size = 1000,
                          learning_rate=def_learning_rate,
                          ExpRep=ExpRep)
    #Pretrained defender
    #defender_agent.model_network.model.load_weights("models_KDD/type_model.h5")    
    
    '''
    Definition for the attacker agent.
    In this case the exploration is better to be greater
    The correlation sould be greater too so gamma bigger
    '''
    attack_valid_actions = list(range(len(env.attack_names)))
    attack_num_actions = len(attack_valid_actions)
	
    att_epsilon = 1
    min_epsilon = 0.82 # min value for exploration

    att_gamma = 0.001
    att_decay_rate = 0.99
    
    att_hidden_layers = 1
    att_hidden_size = 100
    
    att_learning_rate = 0.2
    
    attacker_agent = AttackAgent(attack_valid_actions,obs_size,"EpsilonGreedy",
                          epoch_length = iterations_episode,
                          epsilon = att_epsilon,
                          min_epsilon = min_epsilon,
                          decay_rate = att_decay_rate,
                          gamma = att_gamma,
                          hidden_size=att_hidden_size,
                          hidden_layers=att_hidden_layers,
                          minibatch_size = minibatch_size,
                          mem_size = 1000,
                          learning_rate=att_learning_rate,
                          ExpRep=ExpRep)
    
        
    
    # Statistics
    att_reward_chain = []
    def_reward_chain = []
    att_loss_chain = []
    def_loss_chain = []
    def_total_reward_chain = []
    att_total_reward_chain = []
    
	# Print parameters
    print("-------------------------------------------------------------------------------")
    print("Total epoch: {} | Iterations in epoch: {}"
          "| Minibatch from mem size: {} | Total Samples: {}|".format(num_episodes,
                         iterations_episode,minibatch_size,
                         num_episodes*iterations_episode))
    print("-------------------------------------------------------------------------------")
    print("Dataset shape: {}".format(env.data_shape))
    print("-------------------------------------------------------------------------------")
    print("Attacker parameters: Num_actions={} | gamma={} |" 
          " epsilon={} | ANN hidden size={} | "
          "ANN hidden layers={}|".format(attack_num_actions,
                             att_gamma,att_epsilon, att_hidden_size,
                             att_hidden_layers))
    print("-------------------------------------------------------------------------------")
    print("Defense parameters: Num_actions={} | gamma={} | "
          "epsilon={} | ANN hidden size={} |"
          " ANN hidden layers={}|".format(defender_num_actions,
                              def_gamma,def_epsilon,def_hidden_size,
                              def_hidden_layers))
    print("-------------------------------------------------------------------------------")

        # Main loop
    attacks_by_epoch = []
    attack_labels_list = []
    for epoch in range(num_episodes):
        start_time = time.time()
        att_loss = 0.
        def_loss = 0.
        def_total_reward_by_episode = 0
        att_total_reward_by_episode = 0
        # Reset environment, actualize the data batch with random state/attacks
        states = env.reset()
        
        # Get actions for actual states following the policy
        attack_actions = attacker_agent.act(states)
        states = env.get_states(attack_actions)    
        
        # SARSA: Choose initial defender actions (a)
        defender_actions = defender_agent.act(states)
        
        done = False
       
        attacks_list = []
        # Iteration in one episode
        for i_iteration in range(iterations_episode):
            
            attacks_list.append(attack_actions[0])
            # Apply actions, get rewards and new state
            act_time = time.time()  
            # Environment actuation for these actions
            next_states, def_reward, att_reward, next_attack_actions, done = env.act(defender_actions, attack_actions)
            
            # SARSA: Choose next defender actions (a') using current policy
            next_defender_actions = defender_agent.act(next_states)
            
            # Store experience
            attacker_agent.learn(states, attack_actions, next_states, att_reward, done)
            defender_agent.learn(states, defender_actions, next_states, def_reward, done)
            
            act_end_time = time.time()
            
            # Train network, update loss after at least minibatch_learns
            if ExpRep and epoch * iterations_episode + i_iteration >= minibatch_size:
                def_loss += defender_agent.update_model()
                att_loss += attacker_agent.update_model()
            elif not ExpRep:
                def_loss += defender_agent.update_model()
                att_loss += attacker_agent.update_model()
                
            update_end_time = time.time()

            # Update the state and actions for SARSA
            states = next_states
            attack_actions = next_attack_actions
            defender_actions = next_defender_actions  # SARSA: Carry forward the next actions
            
            # Update statistics
            def_total_reward_by_episode += np.sum(def_reward, dtype=np.int32)
            att_total_reward_by_episode += np.sum(att_reward, dtype=np.int32)
        
        attacks_by_epoch.append(attacks_list)
        # Update user view
        def_reward_chain.append(def_total_reward_by_episode) 
        att_reward_chain.append(att_total_reward_by_episode) 
        def_loss_chain.append(def_loss)
        att_loss_chain.append(att_loss) 

        # Get the test data from the environment
        test_states, test_labels = env_test.get_full()

        # Prepare a new array to hold the mapped labels (5 classes)
        num_samples = test_labels.shape[0]
        num_classes = len(env_test.attack_types)  # should be 5
        mapped_labels = np.zeros((num_samples, num_classes), dtype='float32')

        # For each sample, map the detailed label to one of the 5 main categories
        for i, (idx, row) in enumerate(test_labels.iterrows()):
            detailed_label = row.idxmax()
            main_category = env_test.attack_map[detailed_label]
            class_index = env_test.attack_types.index(main_category)
            mapped_labels[i, class_index] = 1.0

        # Evaluate using the mapped labels
        test_loss = defender_agent.model_network.model.evaluate(test_states, mapped_labels, verbose=0)
        test_loss_chain.append(test_loss)
        
        end_time = time.time()
        print("\r\n|Epoch {:03d}/{:03d}| time: {:2.2f}|\r\n"
                "|Def Loss {:4.4f} | Def Reward in ep {:03d}|\r\n"
                "|Att Loss {:4.4f} | Att Reward in ep {:03d}|"
                .format(epoch, num_episodes, (end_time - start_time), 
                def_loss, def_total_reward_by_episode,
                att_loss, att_total_reward_by_episode))
        
        print("|Def Estimated: {}| Att Labels: {}".format(env.def_estimated_labels,
              env.def_true_labels))
        attack_labels_list.append(env.def_true_labels)
        
if not os.path.exists('models_KDD_'+ismModifDoss):
    os.makedirs('models_KDD_'+ismModifDoss)
# Save trained model weights and architecture, used in test
defender_agent.model_network.model.save_weights("models_KDD_"+ismModifDoss+"/defender_agent_model_"+ismModif+".h5", overwrite=True)
with open("models_KDD_"+ismModifDoss+"/defender_agent_model_"+ismModif+".json", "w") as outfile:
    json.dump(defender_agent.model_network.model.to_json(), outfile)


    
if not os.path.exists('results_KDD_'+ismModifDoss):
    os.makedirs('results_KDD_'+ismModifDoss)    
# Plot training results
plt.figure(1)
plt.subplot(211)
plt.plot(np.arange(len(def_reward_chain)),def_reward_chain,label='Defense')
plt.plot(np.arange(len(att_reward_chain)),att_reward_chain,label='Attack')
plt.title('Total reward by episode')
plt.xlabel('n Episode')
plt.ylabel('Total reward')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
       ncol=2, mode="expand", borderaxespad=0.)

plt.subplot(212)
plt.plot(np.arange(len(def_loss_chain)), def_loss_chain, label='Training Loss (Defender)')
plt.plot(np.arange(len(att_loss_chain)), att_loss_chain, label='Training Loss (Attacker)')
plt.plot(np.arange(len(test_loss_chain)), test_loss_chain, label='Test Loss (Defender)', linestyle='dashed')
plt.title('Loss by episode')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
       ncol=2, mode="expand", borderaxespad=0.)
plt.tight_layout()
#plt.show()
plt.tight_layout()
plt.savefig('results_KDD_'+ismModifDoss+'/loss_comparison_'+ismModif+'.svg', format='svg', dpi=1000)










######################################################################################################################
    
bins = np.arange(len(env.attack_names)) - 0.5  # Ensure correct bin count
# Plot attacks distribution alongside
# Ensure we don't access out-of-range indexes

# MOD-1 Start
valid_epochs = [e for e in [0, 70, 90] if e < len(attacks_by_epoch)]

plt.figure(2, figsize=[12,5])
plt.title("Attacks distribution throughout episodes")

for indx, e in enumerate(valid_epochs):
    plt.subplot(1, len(valid_epochs), indx + 1)  # Adjust for available data
    plt.hist(attacks_by_epoch[e], bins=bins, width=0.9, align='left')
    plt.xlabel(f"Epoch {e}")
    plt.xticks(bins, env.attack_names, rotation=90)

plt.tight_layout()
plt.savefig(f'results_KDD_{ismModifDoss}/Attacks_distribution_{ismModif}.svg', format='svg', dpi=1000)
# MOD-1 Stop

 # Plot attacks distribution alongside -MOD_1

valid_epochs = [e for e in [0,10,20,30,40,60,70,80,90] if e < len(attack_labels_list)]

plt.figure(3, figsize=[10,10])
plt.title("Attacks (mapped) distribution throughout episodes")

for indx, e in enumerate(valid_epochs):
    plt.subplot(3, 3, indx + 1)  # Adjust for available data
    plt.bar(range(5), attack_labels_list[e], tick_label=['Normal','Dos','Probe','R2L','U2R'])
    plt.xlabel(f"Epoch {e}")

plt.tight_layout()
plt.savefig(f'results_KDD_{ismModifDoss}/Attacks_mapped_distribution_{ismModif}.svg', format='svg', dpi=1000)

###########Test########################################################################################################

from tensorflow.keras.models import model_from_json
import itertools
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# Define plot_confusion_matrix function (shared by all sections)
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(f"Normalized confusion matrix - {title}")
    else:
        print(f'Confusion matrix, without normalization - {title}')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Define paths
formated_test_path = "formated_test_adv_" + ismModifDataset + ".data"
formated_train_path = "formated_train_adv_" + ismModifDataset + ".data"  # Added for train data
model_dir = "models_KDD_" + ismModifDoss

# Load the model once (using your original method)
with open(os.path.join(model_dir, "defender_agent_model_" + ismModif + ".json"), "r") as jfile:
    model = model_from_json(json.load(jfile))
model.load_weights(os.path.join(model_dir, "defender_agent_model_" + ismModif + ".h5"))
model.compile(loss=huber_loss, optimizer="Adam")  # Using Adam optimizer as requested

# Define test and train environments
env_test = RLenv('test', formated_test_path=formated_test_path)
env_train = RLenv('train', formated_train_path=formated_train_path)  # Added for train data

# Load test and train data
test_states, test_labels = env_test.get_full()
train_states, train_labels = env_train.get_full()

# Map test labels to main categories and get detailed labels
test_mapped_labels = []
test_detailed_labels = []
for _, label in test_labels.iterrows():
    detailed_label = label.idxmax()
    main_category = env_test.attack_map[detailed_label]
    test_mapped_labels.append(env_test.attack_types.index(main_category))
    test_detailed_labels.append(detailed_label)
test_mapped_labels = np.array(test_mapped_labels)
test_detailed_labels = np.array(test_detailed_labels)

# Predict actions for all test data once
start_time = time.time()
q = model.predict(test_states)
actions = np.argmax(q, axis=1)

# Identify common and exclusive detailed labels
train_attack_names = set(env_train.attack_names)
test_attack_names = set(env_test.attack_names)
common_labels = train_attack_names.intersection(test_attack_names)  # Known labels
exclusive_labels = test_attack_names - train_attack_names          # Unknown labels

# Original test evaluation (kept as-is)
total_reward = 0
true_labels = np.zeros(len(env_test.attack_types), dtype=int)
estimated_labels = np.zeros(len(env_test.attack_types), dtype=int)
estimated_correct_labels = np.zeros(len(env_test.attack_types), dtype=int)

labels, counts = np.unique(test_mapped_labels, return_counts=True)
true_labels[labels] += counts

for indx, a in enumerate(actions):
    estimated_labels[a] += 1
    if a == test_mapped_labels[indx]:
        total_reward += 1
        estimated_correct_labels[a] += 1

action_dummies = pd.get_dummies(actions)
posible_actions = np.arange(len(env_test.attack_types))
for non_existing_action in posible_actions:
    if non_existing_action not in action_dummies.columns:
        action_dummies[non_existing_action] = np.uint8(0)
labels_dummies = pd.get_dummies(test_mapped_labels)

normal_f1_score = f1_score(labels_dummies[0].values, action_dummies[0].values)
dos_f1_score = f1_score(labels_dummies[1].values, action_dummies[1].values)
probe_f1_score = f1_score(labels_dummies[2].values, action_dummies[2].values)
r2l_f1_score = f1_score(labels_dummies[3].values, action_dummies[3].values)
u2r_f1_score = f1_score(labels_dummies[4].values, action_dummies[4].values)

Accuracy = [normal_f1_score, dos_f1_score, probe_f1_score, r2l_f1_score, u2r_f1_score]
Mismatch = estimated_labels - true_labels

acc = float(100 * total_reward / len(test_states))
print('\r\nTotal reward: {} | Number of samples: {} | Accuracy = {:.2f}%'.format(total_reward, len(test_states), acc))
outputs_df = pd.DataFrame(index=env_test.attack_types, columns=["Estimated", "Correct", "Total", "F1_score", "Mismatch"])
for indx, att in enumerate(env_test.attack_types):
    outputs_df.iloc[indx].Estimated = estimated_labels[indx]
    outputs_df.iloc[indx].Correct = estimated_correct_labels[indx]
    outputs_df.iloc[indx].Total = true_labels[indx]
    outputs_df.iloc[indx].F1_score = Accuracy[indx] * 100
    outputs_df.iloc[indx].Mismatch = abs(Mismatch[indx])
print(outputs_df)

# Bar chart (original)
fig, ax = plt.subplots()
fig.set_size_inches(10, 6)
width = 0.35
pos = np.arange(len(true_labels))
plt.bar(pos, estimated_correct_labels, width, color='g', label='Correct estimated')
plt.bar(pos + width, np.abs(estimated_correct_labels - true_labels), width, color='r', label='False negative')
plt.bar(pos + width, np.abs(estimated_labels - estimated_correct_labels), width,
        bottom=np.abs(estimated_correct_labels - true_labels), color='b', label='False positive')
ax.set_xticks(pos + width / 2)
ax.set_xticklabels(env_test.attack_types, rotation='vertical', fontsize='xx-large')
ax.yaxis.set_tick_params(labelsize=15)
plt.legend(('Correct estimated', 'False negative', 'False positive'), fontsize='x-large')
plt.tight_layout()
plt.savefig('results_KDD_' + ismModifDoss + '/test_adv_imp_' + ismModif + '.svg', format='svg', dpi=1000)
plt.close()

# Performance metrics (original)
aggregated_data_test = np.array(test_mapped_labels)
print('Performance measures on Test data')
print('Accuracy =  {:.4f}'.format(accuracy_score(aggregated_data_test, actions)))
print('F1 =  {:.4f}'.format(f1_score(aggregated_data_test, actions, average='weighted')))
print('Precision_score =  {:.4f}'.format(precision_score(aggregated_data_test, actions, average='weighted')))
print('recall_score =  {:.4f}'.format(recall_score(aggregated_data_test, actions, average='weighted')))

# Confusion matrix for all test data (original, renamed for clarity)
cnf_matrix = confusion_matrix(aggregated_data_test, actions)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=env_test.attack_types, normalize=True,
                      title='Normalized Confusion Matrix - All Test Labels')
plt.savefig('results_KDD_' + ismModifDoss + '/confusion_matrix_all_test_' + ismModif + '.svg', format='svg', dpi=1000)
plt.close()

# One vs All metrics (original)
mapa = {0: 'normal', 1: 'DoS', 2: 'Probe', 3: 'R2L', 4: 'U2R'}
yt_app = pd.Series(test_mapped_labels).map(mapa)
perf_per_class = pd.DataFrame(index=range(len(yt_app.unique())), columns=['name', 'acc', 'f1', 'pre', 'rec'])
for i, x in enumerate(pd.Series(yt_app).value_counts().index):
    y_test_hat_check = pd.Series(actions).map(mapa).copy()
    y_test_hat_check[y_test_hat_check != x] = 'OTHER'
    yt_app = pd.Series(test_mapped_labels).map(mapa).copy()
    yt_app[yt_app != x] = 'OTHER'
    ac = accuracy_score(yt_app, y_test_hat_check)
    f1 = f1_score(yt_app, y_test_hat_check, pos_label=x, average='binary')
    pr = precision_score(yt_app, y_test_hat_check, pos_label=x, average='binary')
    re = recall_score(yt_app, y_test_hat_check, pos_label=x, average='binary')
    perf_per_class.iloc[i] = [x, ac, f1, pr, re]
print("\r\nOne vs All metrics: \r\n{}".format(perf_per_class))

# --- Updated Section: Confusion Matrices with Accuracy for Known and Unknown Labels ---

# Function to evaluate and plot confusion matrix with accuracy
def evaluate_and_plot_confusion_matrix(true_labels, predicted_actions, label_subset, subset_name, attack_types, detailed_labels, use_detailed=False):
    # Filter indices based on label subset
    if use_detailed:
        subset_indices = [i for i, d_label in enumerate(detailed_labels) if d_label in label_subset]
    else:
        subset_indices = [i for i, m_label in enumerate(true_labels) if attack_types[m_label] in label_subset]
    
    if not subset_indices:
        print(f"No samples found for {subset_name}")
        return None

    subset_true_labels = true_labels[subset_indices]
    subset_predicted = predicted_actions[subset_indices]

    # Calculate accuracy
    total_reward = np.sum(subset_true_labels == subset_predicted)
    num_samples = len(subset_true_labels)
    acc = float(100 * total_reward / num_samples) if num_samples > 0 else 0.0
    print(f'\n{subset_name} - Total correct: {total_reward} | Number of samples: {num_samples} | Accuracy = {acc:.2f}%')

    # Generate and save confusion matrix
    cnf_matrix = confusion_matrix(subset_true_labels, subset_predicted)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=attack_types, normalize=True,
                          title=f'Normalized Confusion Matrix - {subset_name}')
    cm_plot_path = f'results_KDD_{ismModifDoss}/test_confusion_matrix_{subset_name.lower().replace(" ", "_")}_{ismModif}.svg'
    plt.savefig(cm_plot_path, format='svg', dpi=1000)
    plt.close()
    print(f"Confusion matrix saved to: {cm_plot_path}")
    return acc

# Generate confusion matrices and calculate accuracy for all three cases
acc_all = evaluate_and_plot_confusion_matrix(test_mapped_labels, actions, set(env_test.attack_types), "All Test Labels", env_test.attack_types, test_detailed_labels, use_detailed=False)
acc_known = evaluate_and_plot_confusion_matrix(test_mapped_labels, actions, common_labels, "Known Labels", env_test.attack_types, test_detailed_labels, use_detailed=True)
acc_unknown = evaluate_and_plot_confusion_matrix(test_mapped_labels, actions, exclusive_labels, "Unknown Labels", env_test.attack_types, test_detailed_labels, use_detailed=True)

# Summary of accuracies
print("\nSummary of Accuracies:")
print(f"All Test Labels Accuracy: {acc_all:.2f}%")
print(f"Known Labels Accuracy: {acc_known:.2f}%")
print(f"Unknown Labels Accuracy: {acc_unknown:.2f}%")

######################################################################################################################