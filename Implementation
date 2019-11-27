## Importing packages

library(tidyverse) # metapackage with lots of helpful functions
library(lightgbm) # loading LightGBM
library(pROC) # to use with AUC
library(smotefamily) #create SMOTE dataset
library(RColorBrewer)#used for chart
library(scales) #used for chart

#Define theme for charts

fte_theme <- function() {
 
# Generate the colors for the chart procedurally with RColorBrewer
palette <- brewer.pal("Greys", n=9)
color.background = palette[3]
color.grid.major = palette[3]
color.axis.text = palette[5]
color.axis.title = palette[6]
color.title = palette[8.5]
 
# Begin construction of chart
theme_bw(base_size=9) +
 
# Setting the chart region to a gray color
theme(panel.background=element_rect(fill=color.background, color=color.background)) +
theme(plot.background=element_rect(fill=color.background, color=color.background)) +
theme(panel.border=element_rect(color=color.background)) +

# Formatting the grid
theme(panel.grid.major=element_line(color=color.grid.major,size=.25)) +
theme(panel.grid.minor=element_blank()) +
theme(axis.ticks=element_blank()) +

# Formatting the legend, hide by default
theme(legend.position="none") +
theme(legend.background = element_rect(fill=color.background)) +
theme(legend.text = element_text(size=7,color=color.axis.title)) +

# Setting labels (title and axis) and formatting along with tick marks
theme(plot.title=element_text(color=color.title, size=10, vjust=1.25)) +
theme(axis.text.x=element_text(size=7,color=color.axis.text)) +
theme(axis.text.y=element_text(size=7,color=color.axis.text)) +
theme(axis.title.x=element_text(size=8,color=color.axis.title, vjust=0)) +
theme(axis.title.y=element_text(size=8,color=color.axis.title, vjust=1.25)) +

# Plot margins
theme(plot.margin = unit(c(0.35, 0.2, 0.3, 0.35), "cm"))
}

#Set Seed
set.seed(7464)

#Read Input Data

creditcard <-  read.csv("../input/creditcardfraud/creditcard.csv")


#check data balance proportion

prop.table(table(creditcard$Class))

#Split original data into training & test
train.test.split <- sample(2
                           , nrow(creditcard)
                           , replace = TRUE
                           , prob = c(0.8, 0.2))
train = creditcard[train.test.split == 1,]
test = creditcard[train.test.split == 2,]

#Original Data Model

lgb.train = lgb.Dataset(as.matrix(train[, colnames(train) != "Class"]), label = train$Class)
lgb.test = lgb.Dataset(as.matrix(test[, colnames(test) != "Class"]), label = test$Class)

#Params for original data

params.lgb = list(
     objective = "binary"
    , metric = "auc"
    , min_data_in_leaf = 1
    , min_sum_hessian_in_leaf = 100
    , feature_fraction = 1
    , bagging_fraction = 1
    , bagging_freq = 0
    )

#Model Training

lgb.model <- lgb.train(
        params = params.lgb
        , data = lgb.train
        , valids = list(test = lgb.test)
        , learning_rate = 0.1
        , num_leaves = 7
        , num_threads = 2
        , nrounds = 500
        , early_stopping_rounds = 40
        , eval_freq = 20
        )

print(max(unlist(lgb.model$record_evals[["test"]][["auc"]][["eval"]])))

# get feature importance
lgb.feature.imp = lgb.importance(lgb.model, percentage = TRUE)

# make test predictions
lgb.test.predict = predict(lgb.model, data = as.matrix(test[, colnames(test) != "Class"]), n = lgb.model$best_iter)
auc.lgb = roc(test$Class, lgb.test.predict, plot = TRUE, col = "blue")
print(auc.lgb)
print(lgb.feature.imp)

# Set the number of fraud and legitimate cases, and the desired percentage of legitimate cases
n0 <- nrow(subset(creditcard,Class==0)); n1 <- nrow(subset(creditcard, Class==1)); r0 <- .65

# Calculate the value for the dup_size parameter of SMOTE
ntimes <- ((1 - r0) / r0) * (n0/n1) - 1

# Create synthetic fraud cases with SMOTE
set.seed(1234)
smote_output <- SMOTE(X = creditcard[ , -c(1, 31)], target = creditcard$Class, K = 5, dup_size = ntimes)

#smote output                                                   
credit_smote <- smote_output$data
colnames(credit_smote)[30] <- "Class"
prop.table(table(credit_smote$Class))

# Make a scatter plot of the original and over-sampled dataset

p.original <- ggplot(creditcard, aes(x = V1, y = V2, color = factor(Class))) +
  fte_theme() +
  geom_point() +
  scale_color_manual(values = c('dodgerblue2', 'red'))

p.smote <- ggplot(credit_smote, aes(x = V1, y = V2, color = factor(Class))) +
  fte_theme() +
  geom_point() +
  scale_color_manual(values = c('dodgerblue2', 'red'))
                                                   
p.original; p.smote

#Split SMOTE dataset
set.seed(6474)
smote.train.test.split <- sample(2
                           , nrow(credit_smote)
                           , replace = TRUE
                           , prob = c(0.5, 0.5))
smote.train = creditcard[smote.train.test.split == 1,]
smote.test = creditcard[smote.train.test.split == 2,]


#SMOTE data model

smote.lgb.train <- lgb.Dataset(as.matrix(smote.train[, colnames(smote.train) != "Class"]), label = smote.train$Class)
smote.lgb.test <- lgb.Dataset(as.matrix(smote.test[, colnames(smote.test) != "Class"]), label = smote.test$Class)

#Params for SMOTE data

smote.params.lgb = list(
     objective = "binary"
    , metric = "auc"
    , min_data_in_leaf = 30
    , min_sum_hessian_in_leaf = 100
    , feature_fraction = .9
    , bagging_fraction = 1
    , bagging_freq = 0
    , lambda_l1 = 8
    , lambda_l2 = 1.4
    , min_gain_to_split = 15
    , num_boost_round = 30000
    )

#SMOTE Model Training

smote.lgb.model <- lgb.train(
        params = smote.params.lgb
        , data = smote.lgb.train
        , valids = list(test = smote.lgb.test)
        , learning_rate = 0.088
        , num_leaves = 31
        , num_threads = 2
        , nrounds = 500
        , early_stopping_rounds = 50
        , eval_freq = 20
        )

print(max(unlist(smote.lgb.model$record_evals[["test"]][["auc"]][["eval"]])))

# get feature importance
smote.lgb.feature.imp <- lgb.importance(smote.lgb.model, percentage = TRUE)

# make test predictions
smote.lgb.test.predict = predict(smote.lgb.model, data = as.matrix(smote.test[, colnames(smote.test) != "Class"]), n = smote.lgb.model$best_iter)
smote.auc.lgb = roc(smote.test$Class, smote.lgb.test.predict, plot = TRUE, col = "green")
print(smote.auc.lgb)
print(smote.lgb.feature.imp)

#Original Data Feature 

lgb.feature.imp$Feature <- factor(lgb.feature.imp$Feature, levels=rev(lgb.feature.imp$Feature))

original.plot <- ggplot(lgb.feature.imp, aes(x=Feature, y=Gain)) +
          geom_bar(stat="identity", fill="#34495e", alpha=0.9) +
          geom_text(aes(label=sprintf("%0.1f%%", Gain*100)), color="#34495e", hjust=-0.25, family="Open Sans Condensed Bold", size=2.5) +
          fte_theme() +
          coord_flip() +
          scale_y_continuous(limits = c(0, 0.4)) +
   theme(plot.title=element_text(hjust=0.5), axis.title.y=element_blank()) +
          labs(title="Feature Importance for Credit Card Fraud Detection", y="% of Total Gain in LightGBM Model")

#SMOTE Data Feature Importance
smote.lgb.feature.imp$Feature <- factor(smote.lgb.feature.imp$Feature, levels=rev(smote.lgb.feature.imp$Feature))

smote.plot <- ggplot(smote.lgb.feature.imp, aes(x=Feature, y=Gain)) +
          geom_bar(stat="identity", fill="#34495e", alpha=0.9) +
          geom_text(aes(label=sprintf("%0.1f%%", Gain*100)), color="#34495e", hjust=-0.25, family="Open Sans Condensed Bold", size=2.5) +
          fte_theme() +
          coord_flip() +
          scale_y_continuous(limits = c(0, 0.4)) +
   theme(plot.title=element_text(hjust=0.5), axis.title.y=element_blank()) +
          labs(title="Feature Importance for Credit Card Fraud Detection", y="% of Total Gain in LightGBM Model")


original.plot
smote.plot

# AUC of final iteration on test set
paste("# Rounds:", lgb.model$best_iter)
paste("# Rounds:", smote.lgb.model$best_iter)
paste("AUC of best score - Original Data:", round(lgb.model$best_score*100),2,"%")
paste("AUC of best score - SMOTE Data:", round(smote.lgb.model$best_score*100),2,"%")
