import numpy as np

class LossFunction:
    """Base class for loss functions."""
    def __call__(self, y_true, y_pred):
        """Computes the loss."""
        raise NotImplementedError

    def negative_gradient(self, y_true, y_pred):
        """Computes the negative gradient (pseudo-residuals)."""
        raise NotImplementedError

    def initial_prediction(self, y_true):
        """Computes the initial F_0 prediction."""
        raise NotImplementedError

    def transform_prediction(self, y_pred):
        """Transforms raw model output to the final prediction format (e.g., probabilities)."""
        # Default is no transformation (for regression)
        return y_pred
    
    def working_response(self, y_true, y_pred, mode="grad", hess_clip=1e-6):
        """
        Return (z, w):
          - z: target/pseudo-response for leaf regression
          - w: weights (None means unweighted)
        Default: gradient boosting (z = negative_gradient, w = None)
        """
        z = self.negative_gradient(y_true, y_pred)
        return z, None

class SquaredErrorLoss(LossFunction):
    """Squared Error loss for regression."""
    def __call__(self, y_true, y_pred):
        return 0.5 * np.mean((y_true - y_pred)**2)

    def negative_gradient(self, y_true, y_pred):
        """The negative gradient is simply the residual."""
        return y_true - y_pred

    def initial_prediction(self, y_true):
        """For MSE, the initial prediction is the mean."""
        return np.mean(y_true)
    
    def working_response(self, y_true, y_pred, mode="grad", hess_clip=1e-6):
        # g = y_pred - y_true, negative grad = y_true - y_pred
        z = np.asarray(y_true) - np.asarray(y_pred)
        return z, None

class LogisticLoss(LossFunction):
    """Logistic Loss for binary classification (LogLoss)."""
    def __call__(self, y_true, y_pred_logits):
        """y_true is 0 or 1. y_pred_logits are raw scores."""
        p = 1 / (1 + np.exp(-y_pred_logits))
        return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    def negative_gradient(self, y_true, y_pred_logits):
        """The negative gradient is (y_true - sigmoid(y_pred_logits))."""
        p = 1 / (1 + np.exp(-y_pred_logits))
        return y_true - p

    def initial_prediction(self, y_true):
        """For LogLoss, the initial prediction is the log-odds."""
        p = np.mean(y_true)
        # Avoid infinity with a small epsilon
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return np.log(p / (1 - p))

    def transform_prediction(self, y_pred_logits):
        """Transform logits to probabilities."""
        return 1 / (1 + np.exp(-y_pred_logits))
    
    def working_response(self, y_true, y_pred_logits, mode="grad", hess_clip=1e-6):
        y_true = np.asarray(y_true).reshape(-1)
        raw = np.asarray(y_pred_logits).reshape(-1)

        p = 1.0 / (1.0 + np.exp(-raw))

        if str(mode).lower() == "grad":
            # negative gradient = y - p
            return (y_true - p), None

        if str(mode).lower() == "newton":
            # g = dL/draw = p - y
            g = p - y_true
            # h = d2L/draw2 = p(1-p)
            h = p * (1.0 - p)
            h = np.clip(h, float(hess_clip), np.inf)

            z = -g / h          # pseudo-response
            w = h               # weights
            return z, w

        return (y_true - p), None

# class ExponentialLoss:
#     def _to_pm1(self, y):
#         y = np.asarray(y)
#         if set(np.unique(y)).issubset({0,1}):
#             return 2*y - 1
#         return y

#     def initial_prediction(self, y):
#         return 0.0

#     def negative_gradient(self, y, raw):
#         ypm1 = self._to_pm1(y)
#         return ypm1 * np.exp(-ypm1 * raw)

#     def transform_prediction(self, raw):
#         return 1.0 / (1.0 + np.exp(-2.0 * raw))

#     def __call__(self, y, raw):
#         ypm1 = self._to_pm1(y)
#         return float(np.mean(np.exp(-ypm1 * raw)))

class ExponentialLoss(LossFunction):
    """
    Exponential loss (AdaBoost-style):
      L = mean(exp(-y * raw)), with y in {-1, +1}
    If y is in {0,1}, it will be mapped to {-1,+1}.
    """
    def _to_pm1(self, y):
        y = np.asarray(y).reshape(-1)
        uniq = set(np.unique(y))
        if uniq.issubset({0, 1}):
            return 2 * y - 1
        return y

    def initial_prediction(self, y):
        return 0.0

    def negative_gradient(self, y, raw):
        ypm1 = self._to_pm1(y)
        raw = np.asarray(raw).reshape(-1)
        return ypm1 * np.exp(-ypm1 * raw)

    def transform_prediction(self, raw):
        raw = np.asarray(raw)
        return 1.0 / (1.0 + np.exp(-2.0 * raw))

    def __call__(self, y, raw):
        ypm1 = self._to_pm1(y)
        raw = np.asarray(raw).reshape(-1)
        return float(np.mean(np.exp(-ypm1 * raw)))

    def working_response(self, y, raw, mode="grad", hess_clip=1e-6):
        ypm1 = self._to_pm1(y)
        raw = np.asarray(raw).reshape(-1)

        if str(mode).lower() == "grad":
            return self.negative_gradient(ypm1, raw), None

        if str(mode).lower() == "newton":
            w = np.exp(-ypm1 * raw)
            w = np.clip(w, float(hess_clip), np.inf)
            z = ypm1  # -g/h = y
            return z, w
        
        return self.negative_gradient(ypm1, raw), None

