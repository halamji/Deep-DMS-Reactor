# 通用机器学习模型开发与回归分析框架
# 设置工作目录（根据实际情况修改）
setwd("/D:/working")
library(caret)    # 机器学习框架
library(ggplot2)  # 可视化
library(shapviz)  # SHAP 分析
library(kernelshap)
library(dplyr)    # 数据处理
library(Metrics)  # 模型评估指标

# === 用户自定义区域 ============================================
# 1. 数据加载
train_data <- read.csv('/kaggle/input/chemecompe/482.csv', header = T)

# 2. 指定响应变量名称和特征
response_var <- "dms"  # 目标函数（回归预测）
feature_vars <- c("T", "P", "D", "L", "Catalysis")  # 五个特征

# 确保响应变量是数值类型
train_data[[response_var]] <- as.numeric(train_data[[response_var]])

# 检查数据结构
cat("数据基本信息：\n")
str(train_data)

# 检查响应变量分布
cat("\n响应变量分布：\n")
summary(train_data[[response_var]])

# 3. 设置交叉验证参数
cv_control <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 1,
  savePredictions = "final"  # 保存最终预测结果
)
# 设置可重现结果
set.seed(123)

# 验证数据格式
if(!response_var %in% colnames(train_data)) {
  stop(paste("响应变量", response_var, "不存在于训练数据中"))
}

# 确保所有特征都存在于数据中
missing_features <- setdiff(feature_vars, colnames(train_data))
if(length(missing_features) > 0) {
  stop(paste("以下特征不存在于训练数据中:", paste(missing_features, collapse = ", ")))
}

# 创建包含响应变量和特征的数据集
train_data <- train_data %>% 
  dplyr::select(all_of(c(feature_vars, response_var)))

formula <- as.formula(paste(response_var, "~ ."))

# 定义回归模型列表
model_settings <- data.frame(
  AlgorithmName = c("RandomForest", "GradientBoosting", "SVM", "LinearRegression", "KNN",
                    "PLSRegression", "GBM", "NeuralNet", "ElasticNet", "DecisionTree"),
  Implementation = c("rf", "xgbTree", "svmRadial", "lm", "knn",
                     "pls", "gbm", "nnet", "glmnet", "rpart")
)

# 模型训练和评估
modelContainer <- list()  # 存储训练结果和模型
RMSEresults <- c()        # 存储RMSE结果
R2results <- c()          # 存储R²结果
# 记录总体开始时间
total_start_time <- Sys.time()
cat("===== 开始机器学习模型训练 =====", "\n")
cat("训练模型总数:", nrow(model_settings), "\n")
# 创建存储每个模型训练时间的列表
model_times <- list()

for (idx in seq_len(nrow(model_settings))) {
  # 记录单个模型开始时间
  model_start_time <- Sys.time()
  algoName <- model_settings$AlgorithmName[idx]
  algoImpl <- model_settings$Implementation[idx]
  cat("\n===== 开始训练模型 ", idx, "/", nrow(model_settings), ": ", 
      algoName, " (", algoImpl, ") =====", "\n")
  
  # 根据不同算法应用特定设置
  if (algoName == "NeuralNet") {
    # 神经网络需要额外参数
    trainedModel <- caret::train(formula, data = train_data, method = algoImpl, 
                                 trControl = cv_control, 
                                 metric = "RMSE",
                                 tuneGrid = expand.grid(
                                   size = c(3, 5, 7),
                                   decay = c(0.1, 0.01, 0.001)
                                 ),
                                 trace = FALSE)
  } else if (algoName == "GradientBoosting") {
    # 梯度提升树需要特定参数
    trainedModel <- caret::train(formula, data = train_data, method = algoImpl, 
                                 trControl = cv_control, 
                                 metric = "RMSE",
                                 tuneGrid = expand.grid(
                                   nrounds = c(50, 100),
                                   max_depth = c(3, 5),
                                   eta = c(0.1, 0.3),
                                   gamma = 0,
                                   colsample_bytree = 0.7,
                                   min_child_weight = 1,
                                   subsample = 0.7
                                 ))
  } else if (algoName == "ElasticNet") {
    # ElasticNet需要特定参数
    trainedModel <- caret::train(formula, data = train_data, method = algoImpl, 
                                 trControl = cv_control, 
                                 metric = "RMSE",
                                 tuneGrid = expand.grid(
                                   alpha = seq(0, 1, 0.1),
                                   lambda = c(0.001, 0.01, 0.1, 1)
                                 ))
  } else {
    trainedModel <- caret::train(formula, data = train_data, method = algoImpl, 
                                 trControl = cv_control, 
                                 metric = "RMSE",
                                 tuneLength = 5)
  }
  
  # 计算单个模型训练时间
  model_time <- difftime(Sys.time(), model_start_time, units = "mins")
  model_minutes <- round(as.numeric(model_time), 1)
  # 记录模型训练时间
  model_times[[algoName]] <- model_minutes
  cat("训练完成! 耗时:", model_minutes, "分钟", "\n")
  
  # 保存模型信息
  modelContainer[[algoName]] <- list(
    model = trainedModel, 
    name = algoName,
    impl = algoImpl,
    time = model_minutes
  )
  
  # 获取交叉验证预测结果
  cv_predictions <- trainedModel$pred$pred
  cv_actual <- trainedModel$pred$obs
  
  # 计算评估指标
  rmse <- rmse(cv_actual, cv_predictions)
  r2 <- cor(cv_actual, cv_predictions)^2
  
  # 存储结果
  RMSEresults <- c(RMSEresults, paste0(algoName, ": ", sprintf("%.03f", rmse)))
  R2results <- c(R2results, paste0(algoName, ": ", sprintf("%.03f", r2)))
  
  cat("交叉验证结果: RMSE =", sprintf("%.03f", rmse), "| R² =", sprintf("%.03f", r2), "\n")
  
  # 计算当前进度和预计剩余时间
  if (idx < nrow(model_settings)) {
    elapsed_total <- difftime(Sys.time(), total_start_time, units = "mins")
    avg_time_per_model <- as.numeric(elapsed_total) / idx
    remaining_models <- nrow(model_settings) - idx
    estimated_remaining <- round(avg_time_per_model * remaining_models, 1)
    cat("\n----- 进度报告 -----", "\n")
    cat("已用时间: ", round(as.numeric(elapsed_total), 1), "分钟", "\n")
    cat("平均每模型时间: ", round(avg_time_per_model, 1), "分钟", "\n")
    cat("剩余模型数: ", remaining_models, "\n")
    cat("预计剩余时间: ", estimated_remaining, "分钟", "\n")
    cat("预计结束时间: ", 
        format(Sys.time() + estimated_remaining * 60, "%Y-%m-%d %H:%M:%S"), "\n")
    cat("--------------------", "\n\n")
  }
}

# 计算总耗时
total_time <- difftime(Sys.time(), total_start_time, units = "mins")
total_minutes <- round(as.numeric(total_time), 1)
cat("\n===== 所有模型训练完成! =====", "\n")
cat("总耗时: ", total_minutes, "分钟 (约", round(total_minutes/60, 1), "小时)", "\n")
cat("完成时间: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")

# 输出各模型训练时间详情
cat("\n===== 各模型训练时间统计 =====", "\n")
for (model in names(model_times)) {
  cat(sprintf("%-20s: %5.1f 分钟\n", model, model_times[[model]]))
}
cat("----------------------------", "\n")
cat("最短用时: ", round(min(unlist(model_times)), 1), "分钟 (", 
    names(which.min(model_times)), ")", "\n")
cat("最长用时: ", round(max(unlist(model_times)), 1), "分钟 (", 
    names(which.max(model_times)), ")", "\n")
cat("平均用时: ", round(mean(unlist(model_times)), 1), "分钟", "\n")

# === 可视化模块 ================================================
# 模型性能比较
performance_data <- data.frame(
  Model = model_settings$AlgorithmName,
  RMSE = sapply(modelContainer, function(x) {
    preds <- x$model$pred$pred
    obs <- x$model$pred$obs
    rmse(obs, preds)
  }),
  R2 = sapply(modelContainer, function(x) {
    preds <- x$model$pred$pred
    obs <- x$model$pred$obs
    cor(obs, preds)^2
  })
)

# 按RMSE排序
performance_data <- performance_data[order(performance_data$RMSE), ]

# RMSE比较图
pdf("1_Model_RMSE_Comparison.pdf", width=10, height=6)
ggplot(performance_data, aes(x = reorder(Model, RMSE), y = RMSE, fill = Model)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = sprintf("%.3f", RMSE)), vjust = -0.3, size = 3) +
  labs(title = "模型RMSE比较", x = "模型", y = "RMSE (均方根误差)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  coord_cartesian(ylim = c(0, max(performance_data$RMSE) * 1.1))
dev.off()

# R²比较图
pdf("2_Model_R2_Comparison.pdf", width=10, height=6)
ggplot(performance_data, aes(x = reorder(Model, -R2), y = R2, fill = Model)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = sprintf("%.3f", R2)), vjust = -0.3, size = 3) +
  labs(title = "模型R²比较", x = "模型", y = "R² (决定系数)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  coord_cartesian(ylim = c(0, 1))
dev.off()

# 找出最佳模型
best_model_idx <- which.min(performance_data$RMSE)
best_algo_name <- performance_data$Model[best_model_idx]
best_model_info <- modelContainer[[best_algo_name]]
best_model <- best_model_info$model
best_model_impl <- best_model_info$impl
best_rmse <- performance_data$RMSE[best_model_idx]
best_r2 <- performance_data$R2[best_model_idx]

cat("最优模型:", best_algo_name, "(", best_model_impl, ")", 
    "RMSE:", sprintf("%.03f", best_rmse), "| R²:", sprintf("%.03f", best_r2), "\n")

# === SHAP分析 ==================================================
# 准备SHAP分析数据
# 重新训练最优模型（在整个训练集上）
final_model <- train(
  formula, 
  train_data,
  method = best_model_impl,
  trControl = trainControl(method = "none")  # 无重采样
)

X_train <- train_data %>% 
  dplyr::select(-all_of(response_var)) %>%
  as.data.frame()

# 定义预测函数
pred_fun <- function(object, newdata) {
  predict(object, newdata)
}

# 计算SHAP值
shap_values <- kernelshap(
  final_model, 
  X = X_train,
  pred_fun = pred_fun,
  bg_X = X_train[1:50,]  # 为了速度抽样50个样本
)

# 创建SHAP可视化对象
shap_vis <- shapviz(shap_values, X = X_train)

# 特征重要性排序
feature_importance <- colMeans(abs(shap_values$S))
sorted_features <- names(sort(feature_importance, decreasing = TRUE))

# 可视化设置
visualization_theme <- theme_minimal() + 
  theme(plot.title = element_text(face = "bold", size = 14),
        axis.title = element_text(size = 12))

# 1. 特征重要性柱状图（全局解释）
pdf("3_SHAP_Feature_Importance_Barplot.pdf", width=8, height=6)
sv_importance(shap_vis, kind="bar", show_numbers=TRUE) +
  visualization_theme +
  labs(title = "Feature Importance (Mean Absolute SHAP)",
       subtitle = "Average impact magnitude of each feature on model predictions",
       x = "Mean |SHAP value|", y = "Feature",
       caption = "Bar height indicates feature importance, with value showing mean absolute SHAP")
dev.off()

# 2. 蜂群图（特征效应分布）
pdf("4_SHAP_BeeSwarm_Plot.pdf", width=9, height=7)
sv_importance(shap_vis, kind="bee", show_numbers=TRUE) +
  visualization_theme +
  labs(title = "SHAP Value Distribution (Bee Swarm)",
       subtitle = "Each point represents one sample, color indicates feature value",
       x = "SHAP value (impact on model output)",
       y = "Feature",
       caption = "Red: high feature values | Blue: low feature values | Horizontal spread: direction of effect")
dev.off()

# 3. 特征依赖图（多个特征）
pdf("5_SHAP_Feature_Dependence.pdf", width=10, height=8)
sv_dependence(shap_vis, sorted_features[1:5]) +  # Top 5 important features
  visualization_theme
dev.off()

# 4. 瀑布图（单个样本解释）
pdf("6_SHAP_Sample_Waterfall.pdf", width=9, height=6)
sv_waterfall(shap_vis, row_id=2) +  # Explaining sample #2
  labs(title = "Prediction Breakdown (Waterfall Plot)",
       subtitle = "Cumulative contribution of features to final prediction",
       caption = "E[f(x)] = base value | f(x) = model output\nBar length: feature contribution | Direction: sign of effect")
dev.off()

# 5. 部分依赖图（PDP）
pdf("7_SHAP_Partial_Dependence.pdf", width=10, height=8)
par(mfrow = c(2, 3))  # 2x3网格
for (feature in sorted_features[1:5]) {  # 前5个重要特征
  pdp_data <- shapviz::pdp(
    x = final_model,
    X = X_train,
    pred_wrapper = pred_fun,
    feature = feature,
    grid.size = 50
  )
  
  plot(pdp_data, type = "l", main = paste("Partial Dependence -", feature),
       xlab = feature, ylab = "Predicted dms", col = "blue", lwd = 2)
}
dev.off()

# 保存最佳模型
saveRDS(best_model, "best_regression_model.rds")
cat("\n最佳模型已保存为: best_regression_model.rds\n")