from dgllife.utils import Meter
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score, f1_score, precision_score, recall_score,r2_score
import logging
logger = logging.getLogger()

class MoreMeter(Meter):
    """
    Calculates the precision and recall.
    - `update` must receive output of the form `(y_pred, y)`.
    - `y_pred` must be in the following shape (batch_size, num_categories, ...)
    - `y` must be in the following shape (batch_size, ...)
    """

    def __init__(self, mean=None, std=None):
        super().__init__(mean, std)

    def acc(self, reduction='none'):
        # logger.info('acc be:',self.y_pred[0])
        for i in range(len(self.y_pred)):
            self.y_pred[i] = self.y_pred[i].round()
        # logger.info('acc af:',self.y_pred[0])
        return self.multilabel_score(accuracy_score, reduction)

    def f1(self, reduction='none'):
        return self.multilabel_score(f1_score, reduction)

    def precision(self, reduction='none'):
        return self.multilabel_score(precision_score, reduction)

    def recall(self, reduction='none'):
        return self.multilabel_score(recall_score, reduction)

    def mse(self,reduction='none'):
        return self.multilabel_score(mean_squared_error, reduction)

    def r2(self,reduction='none'):
        return self.multilabel_score(r2_score, reduction)
