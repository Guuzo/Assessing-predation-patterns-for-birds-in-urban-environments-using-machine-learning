This repository contains an object detector for detecting green caterpillars, other caterpillars and non-caterpillars created using YOLOv8 by Ultralytics.

The object detector can be used with "best.pt" for the best performing model and "last.pt" for the last iteration of the model.

"YOLO_training_script.py" in combination with "configs.yaml"can be used to train a new model or the retrain the existing model.

"YOLO_predicting_sorting.py" can be used to apply the YOLO model to a dataset, the annotations are sorted by categoryin separate folders.

"F1_curve.PNG", "P_curve.PNG", "PR_curve.PNG" and "R_curve.PNG" shows curves based on the performance of the object detector.

confusion_matrix.png" and "confusion_matrix_normalized.png" show are confusion matrix and a normalized confusion matrix on the validation dataset.

"labels.jpg" shows some metrics on the labels used in the training process.

"results.png" shows results on the models performance and training process.

"val_batch0_labels.jpg", "val_batch0_pred.jpg", "val_batch1_labels.jpg", "val_batch1_pred.jpg", "val_batch2_labels.jpg" and "val_batch2_pred.jpg" shows batches of validation data with the manually annotated labels and labels predicted by the model, this can be used to get a quick overview of the models behaviour.
