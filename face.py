import numpy as np

class Face(object):
    def __init__(self):
        self.training = np.loadtxt("svm.train.normgrey")
        np.random.shuffle(self.training)
        self.test = np.loadtxt("svm.test.normgrey")
        self.t_pos = 0


        self.test_data = self.test[:,:-1]
        self.test_labels_tmp= self.test[:,-1]
        self.test_labels_tmp[self.test_labels_tmp == -1] = 0
        self.test_labels = np.zeros((2,np.size(self.test_labels_tmp)))
        self.test_labels[self.test_labels_tmp.astype(int),np.arange(np.size(self.test_labels_tmp))] = 1
        self.test_labels = self.test_labels.T
        #print self.test_labels

        self.training_xs = self.training[:,:-1]
        self.ys_tmp  = self.training[:,-1]
        self.ys_tmp[self.ys_tmp == -1] = 0

        self.training_ys = np.zeros((2,np.size(self.ys_tmp)))
        #print( self.ys_tmp.astype(int))
        self.training_ys[self.ys_tmp.astype(int),np.arange(np.size(self.ys_tmp))] = 1
        self.training_ys = self.training_ys.T
        #print( self.training_ys)



    def next_batch(self,x):
        assert(x>0)
        #print(str(np.size(self.training_xs,1)) + " " + str(np.size(self.training_ys)) )
        #print(self.t_pos)
        if(self.t_pos+x<=np.size(self.test_data,0)):
            xs =  self.training_xs[self.t_pos:self.t_pos+x,:]
            ys =  self.training_ys[self.t_pos:self.t_pos+x]
            self.t_pos +=x
        else:
            xs =  self.training_xs[self.t_pos:,:]
            ys =  self.training_ys[self.t_pos:]
            self.t_pos = 0

        #print(xs.shape)
        #print(ys.shape)
        return (xs,ys)

def make_Face():
    face = Face()
    return face



