# **ML Hello World**

A simple machine learning project that implements linear regression from scratch to understand the fundamentals of ML.

## **Description**

This project demonstrates the core concepts of machine learning by building a linear regression model without using ML frameworks. The model learns to predict the relationship `y = 4x + 5` through gradient descent and backpropagation.

**Key Features:**
- Custom implementation of feedforward pass
- Manual gradient calculation using calculus
- Stochastic Gradient Descent (SGD) for weight updates
- Mean Squared Error (MSE) loss function
- Training and testing data evaluation

**What it learns:** Given input values, the model learns the weight (≈4) and bias (≈5) to accurately predict outputs following the linear relationship.

## **Key Equations Used**

**1. Prediction Function (Feedforward):**
$$\hat{y} = wx + b$$
Where $w$ is the weight, $x$ is the input, and $b$ is the bias.

**2. Loss Function (Mean Squared Error):**
$$L = \frac{1}{2}(\hat{y} - y)^2$$
Measures the squared difference between predicted ($\hat{y}$) and actual ($y$) values.

**3. Gradient with respect to Weight:**
$$\frac{\partial L}{\partial w} = (\hat{y} - y) \cdot x$$
Indicates how much to adjust the weight to reduce loss.

**4. Gradient with respect to Bias:**
$$\frac{\partial L}{\partial b} = (\hat{y} - y)$$
Indicates how much to adjust the bias to reduce loss.

**5. Weight Update Rule (Gradient Descent):**
$$w_{\text{new}} = w - \alpha \cdot \frac{\partial L}{\partial w}$$
Where $\alpha$ is the learning rate (0.01).

**6. Bias Update Rule (Gradient Descent):**
$$b_{\text{new}} = b - \alpha \cdot \frac{\partial L}{\partial b}$$
Updates bias in the direction that reduces loss.

## **Training Output:**

```
Epoch 0, Loss: 608.3955250874391
Epoch 1, Loss: 265.53303519599353
Epoch 2, Loss: 162.16648087375367
Epoch 3, Loss: 101.17572216752896
Epoch 4, Loss: 63.22281696324665
Epoch 5, Loss: 39.51141534335085
Epoch 6, Loss: 24.693075349154856
Epoch 7, Loss: 15.432207916351006
Epoch 8, Loss: 9.644527858200968
Epoch 9, Loss: 6.027453651631055
Epoch 10, Loss: 3.766923385675324
Epoch 11, Loss: 2.3541801586495974
Epoch 12, Loss: 1.4712707565181211
Epoch 13, Loss: 0.9194868247582829
Epoch 14, Loss: 0.5746433939222115
Epoch 15, Loss: 0.35912970288100426
Epoch 16, Loss: 0.22444205372498593
Epoch 17, Loss: 0.14026752751492758
Epoch 18, Loss: 0.08766173249893551
Epoch 19, Loss: 0.054785162901634814
Epoch 20, Loss: 0.034238589503066234
Epoch 21, Loss: 0.02139778270376414
Epoch 22, Loss: 0.013372779407180095
Epoch 23, Loss: 0.008357465422884437
Epoch 24, Loss: 0.0052230898430285855
Epoch 25, Loss: 0.003264227385691694
Epoch 26, Loss: 0.0020400147701309258
Epoch 27, Loss: 0.0012749296450959907
Epoch 28, Loss: 0.0007967812898923892
Epoch 29, Loss: 0.0004979572216903504
Epoch 30, Loss: 0.0003112038369613787
Epoch 31, Loss: 0.0001944902572368053
Epoch 32, Loss: 0.00012154882320662667
Epoch 33, Loss: 7.596327257119305e-05
Epoch 34, Loss: 4.747408183388227e-05
Epoch 35, Loss: 2.9669449059842517e-05
Epoch 36, Loss: 1.85422481806894e-05
Epoch 37, Loss: 1.1588181732023337e-05
Epoch 38, Loss: 7.2421614976692054e-06
Epoch 39, Loss: 4.526068400654257e-06
Epoch 40, Loss: 2.828616176815011e-06
Epoch 41, Loss: 1.7677747588990488e-06
Epoch 42, Loss: 1.1047902588610535e-06
Epoch 43, Loss: 6.904508110718813e-07
Epoch 44, Loss: 4.315048206549116e-07
Epoch 45, Loss: 2.6967367879194585e-07
Epoch 46, Loss: 1.685355285796916e-07
Epoch 47, Loss: 1.0532813035747708e-07
Epoch 48, Loss: 6.58259723520931e-08
Epoch 49, Loss: 4.113866467846496e-08
Epoch 50, Loss: 2.571006049885694e-08
Epoch 51, Loss: 1.60677847959882e-08
Epoch 52, Loss: 1.0041738651886066e-08
Epoch 53, Loss: 6.2756949032559274e-09
Epoch 54, Loss: 3.922064483534067e-09
Epoch 55, Loss: 2.4511372924708894e-09
Epoch 56, Loss: 1.5318651825942796e-09
Epoch 57, Loss: 9.573559771383095e-10
Epoch 58, Loss: 5.983101367788876e-10
Epoch 59, Loss: 3.7392049391933265e-10
Epoch 60, Loss: 2.336857211120225e-10
Epoch 61, Loss: 1.4604445897940417e-10
Epoch 62, Loss: 9.127208929927317e-11
Epoch 63, Loss: 5.7041495050872856e-11
Epoch 64, Loss: 3.5648709083705326e-11
Epoch 65, Loss: 2.227905242309191e-11
Epoch 66, Loss: 1.3923538605087525e-11
Epoch 67, Loss: 8.701668433336933e-12
Epoch 68, Loss: 5.4382032954521774e-12
Epoch 69, Loss: 3.3986649002883928e-12
Epoch 70, Loss: 2.124032975753105e-12
Epoch 71, Loss: 1.3274377469237241e-12
Epoch 72, Loss: 8.295968041053177e-13
Epoch 73, Loss: 5.184656398738044e-13
Epoch 74, Loss: 3.2402079893395713e-13
Epoch 75, Loss: 2.0250035957591172e-13
Epoch 76, Loss: 1.2655482544575095e-13
Epoch 77, Loss: 7.909182803283346e-14
Epoch 78, Loss: 4.9429306807015614e-14
Epoch 79, Loss: 3.089138821766622e-14
Epoch 80, Loss: 1.930591195856857e-14
Epoch 81, Loss: 1.2065442117917304e-14
Epoch 82, Loss: 7.54043077684046e-15
Epoch 83, Loss: 4.712475109558817e-15
Epoch 84, Loss: 2.9451130502714755e-15
Epoch 85, Loss: 1.8405807691064773e-15
Epoch 86, Loss: 1.1502910638008665e-15
Epoch 87, Loss: 7.188870346550304e-16
Epoch 88, Loss: 4.492763390742634e-16
Epoch 89, Loss: 2.8078020625634847e-16
Epoch 90, Loss: 1.7547663549173534e-16
Epoch 91, Loss: 1.0966606682985345e-16
Epoch 92, Loss: 6.853700026875647e-17
Epoch 93, Loss: 4.2832977014050174e-17
Epoch 94, Loss: 2.6768926275930235e-17
Epoch 95, Loss: 1.6729541310535006e-17
Epoch 96, Loss: 1.045530230637183e-17
Epoch 97, Loss: 6.534154144235809e-18
Epoch 98, Loss: 4.083589696659983e-18
Epoch 99, Loss: 2.552086350053823e-18
Trained weight: 4.0000000001585, Trained bias: 4.9999999987317

```

## **Testing the Model**

```
Testing the trained model:
Input: 3, Predicted: 16.999999999207198, Correct: 17
Input: 4, Predicted: 20.9999999993657, Correct: 21
Input: 5, Predicted: 24.9999999995242, Correct: 25
```

# **Discussion**
If you reached till here, then I would like to thank you for your interests in this humble beginner project.

As you can see, the ended up with $$W \approx 4 $$ 
$$B \approx 5$$
Which is very close to what the real equation was
$$f(x) = 4x + 5 $$