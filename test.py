import numpy as np
data = np.loadtxt("./data.txt")
data1 = np.array(data.T)
# print(data)
#len(data[1]) 15
weight = np.array([0.5]*15)
#print(weight)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx*(1-fx)
def mse_loss(y_true,y_pred):
    return ((y_true-y_pred)**2).mean()

class OurNeurlNetwork():
    def __init__(self):
        self.w1=np.random.normal()
        self.w2=np.random.normal()
        self.w3=np.random.normal()
        self.w4=np.random.normal()
        self.w5=np.random.normal()
        self.w6=np.random.normal()
        self.b1=np.random.normal()
        self.b2=np.random.normal()
        self.b3=np.random.normal()

    def feedforward(self,x):
        h1 = sigmoid(self.w1*x[0]+self.w2*x[1]+self.b1)
        h2 = sigmoid(self.w5*x[0]+self.w4*x[1]+self.b2)
        o1 = sigmoid(self.w5*h1+self.w6*h2+self.b3)
        return o1

    def train(self,data,all_y_trues):
        learn_rate = 0.1
        epochs = 1000
        for epoch in range(epochs):
            for x,y_true in zip(data,all_y_trues):
                sum_h1 = self.w1*x[0]+self.w2*x[1]+self.b1
                h1 = sigmoid(sum_h1)
                sum_h2 = self.w5*x[0]+self.w4*x[1]+self.b2
                h2 = sigmoid(sum_h2)
                sum_o1 = self.w5*h1+self.w6*h2+self.b3
                o1 = sigmoid(sum_o1)

                y_pred = o1
                d_L_y_pred = -2*(y_true-y_pred)
                d_ypred_d_w5 = h1*deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2*deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)
                d_ypred_d_h1 = self.w5*deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6*deriv_sigmoid(sum_o1)

                d_h1_d_w1 = x[0]*deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1]*deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h2)
                d_h2_d_w3 = x[0]*deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1]*deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)
                self.w1 -=learn_rate*d_L_y_pred*d_ypred_d_h1*d_h1_d_w1
                self.w2 -=learn_rate*d_L_y_pred*d_ypred_d_h1*d_h1_d_w2
                self.b1-=learn_rate*d_L_y_pred*d_ypred_d_h1*d_h1_d_b1
                self.w3-=learn_rate*d_L_y_pred*d_ypred_d_h2*d_h2_d_w3
                self.w4-=learn_rate*d_L_y_pred*d_ypred_d_h2*d_h2_d_w4
                self.b2-=learn_rate*d_L_y_pred*d_ypred_d_h2*d_h2_d_b2
                if epoch %10 == 0 :
                    y_preds = np.apply_along_axis(self.feedforward,1,data)
                    loss = mse_loss(all_y_trues,y_preds)
                    print("Epoch %d loss:%.3f"%(epoch,loss))

def detect(x,y):
    data = np.array([x,y])
    score = network.feedforward(data)
    if(score<=0.5):
        print("该样本的预测值：%.3f,预测的种类：apf"%score)
    else:
        print("该样本的预测值：%.3f,预测的种类：af"%score)
if __name__ == '__main__':
    # data=np.array(
    #     [
    #         [-2,-1],
    #         [25,6],
    #         [17,4],
    #         [-15,-6],
    #     ]
    # )
    # all_y_trues = np.array([
    #     1,
    #     0,
    #     0,
    #     1,
    # ])
    data = np.loadtxt("./data.txt")
    data1 = np.array(data)
    all_y_trues = np.array([
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ])
    network = OurNeurlNetwork()
    network.train(data,all_y_trues)
    #预测
    # ceshi1 = np.array([1.24,1.80])
    # ceshi2= np.array([1.28,1.84])
    # ceshi3=np.array([1.40,2.04])
    # score = network.feedforward(ceshi1)
    # if(score<=0.5):
    #     print("ceshi1的种类：apf")
    # else:
    #     print("ceshi1的种类:af")
    # print("ceshi1:%.3f"%network.feedforward(ceshi1))
    # print("ceshi2:%.3f"%network.feedforward(ceshi2))
    # print("ceshi3:%.3f"%network.feedforward(ceshi3))
    #if():
    detect(1.24, 1.80)
    detect(1.28,1.84)
    detect(1.40,2.04)

