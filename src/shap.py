import shap
import numpy as np
import matplotlib.pyplot as plt

class ShapExplainer:
    def __init__(self, model, x_train, x_test):
        self.model = model
        self.x_test = x_test
        self.x_train = x_train
        self.validate_inputs()
        self.init_explainer()
    def validate_inputs(self):
        if self.model is None:
            raise ValueError('Model is None.Train the model before SHAP analysis.')
        if self.x_train is None or self.x_test is None:
            raise ValueError('Train or test data is missing.')
        if not hasattr(self.model, 'predict'):
            raise TypeError('Model must implement predict().')
        def init_explainer(self):
            # Use TreeExplainer for RandomForest
            self.explainer = shap.TreeExplainer(self.model)
            raw_shap = self.explainer.shap_values(self.x_test)
            if isinstance(self.explainer.expected_value, list):
                self.expected_value = self.explainer.expected_value[1]
            else:
                self.expected_value = self.explainer.expected_value
            # Normalize SHAP values -> always (n_samples, n_features)
            if isinstance(raw_shap, list):
                self.shap_values = raw_shap[1]
            elif raw_shap.ndim==3:
                self.shap_values = raw_shap[:,:,1]

            else:
                self.shap_values = raw_shap
            # Safety check
            if self.shap_values.shape[1] != self.x_test.shape[1]:
                raise ValueError(
                    f'SHAP shape {self.shap_values.shape} does not match X_test {self.x_test.shape}'
                )
        def plot_shap_summary(self):
            try:
                shap.summary_plot(self.shap_values,self.x_test, plot_type='bar',show=True )
            except Exception as e:
                raise RuntimeError(f'failed to plot SHAP summary: {e}')
        def plot_force(self, index):
            try:
                row = (self.x_test.iloc[index]
                       .astype(float)
                       .round(2)
                       .values)
                shap.force_plot(
                    self.expected_value,
                    self.shap_values[index],
                    row,
                    feature_names=self.x_test.columns,
                    matplotlib=True,
                    show=True
                )
            except IndexError:
                print(f'Index {index} is out of bounds.')
            except Exception as e:
                print(f'Error generating SHAP force plot: {e}')

                
