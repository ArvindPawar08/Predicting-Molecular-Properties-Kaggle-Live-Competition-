library(dplyr)
library(readr)
library(xgboost)
library(caret)
library(gridExtra)

#Import train data
train_data <- read_csv("/Users/abhishek/Downloads/input/train.csv") 

#Import Test Data
test_data <- read_csv("/Users/abhishek/Downloads/input/test.csv" )

#Assign Target variable to a temp variable so that we can combine test and train 
#to do our feature engineering.
y <- as.matrix(train_data$scalar_coupling_constant)

#Remove target variable from train dataset.
train_data$scalar_coupling_constant <- NULL
#Combine train and test data
train_test <- rbind(train_data,test_data)

#Types of molecule
coupling_type <- train_test$type
types <- unique(coupling_type)

#Calculate the mean of coupling contribution for a given molecule 
#group by csum, atom_index_0,atom_index_1,type.
grp_scc_by_csum_a1_a2 <- read_csv("../input/scalar_coupling_contributions.csv") %>% 
  left_join(read_csv("../input/scalar_coupling_contributions.csv") %>% 
              group_by(molecule_name) %>% 
              summarise(csum = n())) %>% 
  select(-molecule_name) %>% 
  group_by(csum, atom_index_0,atom_index_1,type) %>% 
  summarise_all(funs(mean, sum, .args = list(na.rm = TRUE)))%>% 
  select(-fc_sum, -sd_sum, -pso_sum, -dso_sum)



#Join the train and test with structure based join based on atom index 1 and atom index 2 
#and calculate the distance combining the coordinate.
train_test <- train_test %>% 
  left_join(read_csv("../input/structures.csv"), 
            by = c("molecule_name","atom_index_0" = "atom_index")) %>% 
  left_join(read_csv("../input/structures.csv"), 
            by = c("molecule_name","atom_index_1" = "atom_index")) %>%
  mutate(
    HH = 1*(atom.x =="H" & atom.y =="H"),
    HC = 1*(atom.x =="H" & atom.y =="C"),
    HN = 1*(atom.x =="H" & atom.y =="N"),
    x_dist = x.x - x.y, 
    y_dist = y.x - y.y,  
    z_dist = z.x - z.y, 
    dist = sqrt(x_dist^2 + y_dist^2 + z_dist^2)) %>% 
  select(-id, -atom.x, -atom.y)


#Mean of x,x,z and distance, grouping by molecule
grp_by_mol <- train_test  %>% 
  select(-type) %>% 
  group_by(molecule_name) %>%
  summarise_all(funs( mean, sum,  .args = list(na.rm = TRUE)) ) %>%
  select( -atom_index_0_mean, -atom_index_1_mean, -x.x_mean, -y.x_mean,
          -z.x_mean, -x.y_mean, -y.y_mean, -z.y_mean, -HH_mean, -HC_mean,
          -HN_mean, -atom_index_0_sum, -atom_index_1_sum, -x.x_sum, -y.x_sum,
          -z.x_sum, -x.y_sum, -y.y_sum, -z.y_sum, -HH_sum, -HC_sum, -HN_sum,
          -x_dist_sum, -y_dist_sum, -z_dist_sum, -dist_sum)


# Calculate the difference between mean of distance and dist grouping by type
grp_dist_by_type <- train_test %>% 
  left_join(
    train_test %>% 
      group_by(type) %>% 
      summarise(dist_type = mean(dist)
      ), by = "type") %>% 
  mutate(
    dist_dif = dist_type - dist
)

#Combining all the feature and do feature scaling.
train_test <- train_test %>% 
  left_join(train_test %>% group_by(molecule_name) %>% summarise(csum = n())) %>% 
  left_join(grp_scc_by_csum_a1_a2) %>%  
  left_join(grp_by_mol) %>% 
  left_join(grp_dist_by_type) %>% 
  mutate(C1 = 1*(type == "1JHC"),
         N1 = 1*(type == "1JHN"), 
         C2 = 1*(type == "2JHC"), 
         H2 = 1*(type == "2JHH"), 
         N2 = 1*(type == "2JHN"),
         C3 = 1*(type == "3JHC"), 
         H3 = 1*(type == "3JHH"), 
         N3 = 1*(type == "3JHN")) %>% 
  select(-molecule_name, -x.x, -y.x, -z.x, -x.y, -y.y, -z.y, -csum) 

names(train_test)




                        #########################################
                                  #Creating the sample
                        #########################################


tri <- 1:4658147
df_train <- train_test[tri,]
df_test <- train_test[-tri,]
target <- y
train_sample <- cbind(df_train,target)

train_sample <- train_sample %>% group_by(type) %>% sample_n(40000)
train_sample <- train_sample[sample(1:320000), ]
target <- as.matrix(train_sample$target)
train_sample$target <- NULL
train_sample$type <- NULL
df_test$type <- NULL

                        
                        #########################################
                               #Set the boostting Parameter
                        #########################################


param_tree <- list(booster = "gbtree"
              , objective = "reg:linear"
              , subsample = 0.7
              , max_depth = 6
              , colsample_bytree = 0.5
              , eta = 0.05
              , eval_metric = 'mae'
              , base_score = mean(target)
              , min_child_weight = 100)

param_linear <- list(booster = "gblinear"
              , objective = "reg:linear"
              , feature_selector = "greedy"
              , updater = "coord_descent"
              , eval_metric = 'mae'
              , base_score = mean(y)
              , min_child_weight = 100)

                          #########################################
                                     #Cross Validation
                          #########################################

foldsCV <- createFolds(target, k=7, list=TRUE, returnTrain=FALSE)
df_train <- xgb.DMatrix(data=as.matrix(train_sample),label=target, missing=NA)
df_test <- xgb.DMatrix(data=as.matrix(df_test), missing=NA)
feature_names <- names(train_sample)
xgb_cv <- xgb.cv(data=df_train,
                 params=param_tree, #Need to change the param for linear
                 nrounds=100,
                 prediction=TRUE,
                 maximize=FALSE,
                 folds=foldsCV,
                 early_stopping_rounds = 30,
                 print_every_n = 1
)

# Check best results and get best nrounds
print(xgb_cv$evaluation_log[which.min(xgb_cv$evaluation_log$test_mae_mean)])
nrounds <- xgb_cv$best_iteration

                                      ################
                                        # Final model
                                      ################

xgb <- xgb.train(params = param_tree
                 , data = df_train
                 , nrounds = nrounds
                 , verbose = 1
                 , print_every_n = 5
)

                                ##############################
                                    # Result Prediction
                                ##############################

# Feature Importance
importance_matrix <- xgb.importance(feature_names,model=xgb)

# Predict
preds <- predict(xgb,df_test)


