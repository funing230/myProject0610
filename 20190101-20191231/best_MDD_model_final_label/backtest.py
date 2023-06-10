import matplotlib.pyplot as plt
import pandas as pd
from util import get_data_raw,train_test_split,\
    lgbm_opt,svm_opt,sgd_opt,gpc_opt,gnb_opt,dtc_opt,ada_opt,gbc_opt,lgbm_opt\
    ,xgb_opt,cat_opt,ridge_opt,mlp_opt,Mlp_opt,knn_opt,gbc_ada_opt_test

n_steps = 5
X, y = get_data_raw()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,shuffle=False,stratify=None)

# clf, y_pred=lgbm_opt(X_train, y_train, X_test, y_test)     #0.4850631578947368
# clf, y_pred=svm_opt(X_train, y_train, X_test, y_test)      #0.5013440860215054
# clf, y_pred=sgd_opt(X_train, y_train, X_test, y_test)     #0.5
# clf, y_pred=gpc_opt(X_train, y_train, X_test, y_test)     #0.5068576653459311

# clf, y_pred=gnb_opt(X_train, y_train, X_test, y_test)  # Accuracy: 0.5069306930693069 0.6273503216229589
# clf, y_pred=dtc_opt(X_train, y_train, X_test, y_test)    #0.6247026169706582
# clf, y_pred=ada_opt(X_train, y_train, X_test, y_test)     # Accuracy: 0.7683168316831683 best----------------------0.6837837837837838
# clf, y_pred=gbc_opt(X_train, y_train, X_test, y_test)       #best--------- Accuracy: 0.8198019801980198            0.5522606045212091
# clf, y_pred=lgbm_opt(X_train, y_train, X_test, y_test)      #0.4921721721721722
# clf, y_pred=xgb_opt(X_train, y_train, X_test, y_test)      #0.4318461538461538
# clf, y_pred=cat_opt(X_train, y_train, X_test, y_test)      #0.5492172211350294
# clf, y_pred=ridge_opt(X_train, y_train, X_test, y_test)      # 0.4534368299521689
# clf, y_pred=mlp_opt(X_train, y_train, X_test, y_test)      # 0.5059306198716387

# y_pred=Mlp_opt(X_train, y_train, X_test, y_test)      #  Accuracy: 0.46534653465346537  0.502687164104487
# y_pred=knn_opt(X_train, y_train, X_test, y_test)    # Accuracy: 0.504950495049505


y_pred=gbc_ada_opt_test(X_train, y_train, X_test, y_test)  #best--best--best--best--------- Accuracy: 0.8198019801980198

# from sklearn.metrics import roc_curve
# from sklearn.metrics import auc
# fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred, pos_label=1)
# plt.plot(fpr,tpr,marker = '.')
# plt.show()
#
# AUC = auc(fpr, tpr)

pred=pd.Series(y_pred)
test=pd.Series(y_test)

temp=(pred == test).sum()

temp1=temp.item()

print("Accuracy:", (y_pred == y_test).sum().item() / len(y_test))

unio=pd.concat([pd.DataFrame(y_pred),pd.DataFrame(y_test)],ignore_index=True,axis=1)
unio.columns = ['y_pred','y_test']
unio.index=X_test.index
pd.DataFrame(unio).to_csv("best_MDD_machine_learning_for_lable_20190101_20191231.csv", index=True)
#
# print('-------------AUC-------------',AUC)

print()

