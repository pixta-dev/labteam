import numpy as np
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrecisionRecallFk:
    """
    Calculate precision, recall, f1 for predictions.
    """

    def __init__(self, enable_logger=False, threshold=0.5, eps=1e-9):
        if enable_logger:
            global logger
            self.logger = logger

        self.threshold = threshold
        self.eps = eps

    def __call__(self,
                 prediction: np.array,
                 ground_truth: np.array,
                 betas: list = [1],
                 top_ks: list = None) -> (float, float, float):
        """
        compute F-beta score

        input:
            + prediction: predictions of model, np.array of shape [B, N] with B be the batchsize
            and N is the number of classes
            + ground_truth: self explanatory, must have the same shape as prediction
            + betas: a list of betas
            + top_ks: if specified, compute f_score of top_k most confident prediction
        output:
            + f_score: (1+beta**2) * precision*recall/(beta**2 * precision+recall)
        """
        if top_ks is None:
            return self.f_k_score(prediction,
                                  ground_truth,
                                  betas)
        else:
            return self.f_k_score_top(prediction,
                                      ground_truth,
                                      betas,
                                      top_ks)

    def f_k_score(self,
                  prediction: np.array,
                  ground_truth: np.array,
                  betas: list = [1],
                  threshold: float = None):
        """
        compute F-beta score

        input:
            + prediction: unthresholded output of the model, np.array of shape [B, N] with B be the 
            batchsize and N is the number of classes
            + ground_truth: self explanatory, must have the same shape as prediction
            + betas: a list of betas
        output:
            + f_scores: (1+beta**2) * precision*recall/(beta**2 * precision+recall)
        """
        assert prediction.shape == ground_truth.shape

        if threshold is None:
            prediction = prediction >= self.threshold
        else:
            prediction = (prediction >= threshold)

        prediction = prediction.astype(int)

        ground_truth = ground_truth.reshape(prediction.shape)
        num_prediction = np.count_nonzero(prediction, axis=1)
        num_ground_truth = np.count_nonzero(ground_truth, axis=1)

        if hasattr(self, "logger"):
            self.logger.info(
                "Predictions per item: {}, Labels per item: {}".format(np.mean(num_prediction),
                                                                       np.mean(num_ground_truth))
            )

        num_true_positive_pred = np.count_nonzero(
            ground_truth & prediction, axis=1)

        precision = num_true_positive_pred/num_prediction + self.eps
        recall = num_true_positive_pred/num_ground_truth + self.eps

        f_scores = {}
        for beta in betas:
            beta_squared = beta ** 2
            f_score = np.nan_to_num(
                (1 + beta_squared)*precision*recall / (beta_squared * precision+recall))
            f_scores["F{}".format(beta)] = np.nanmean(f_score)

        if hasattr(self, "logger"):
            self.logger.info(
                "Can't give predictions to {} items".format(
                    np.count_nonzero(np.isnan(precision)))
            )

        return {"precision": np.nanmean(precision), "recall": np.nanmean(recall), "f_score": f_scores}

    def f_k_score_top(self,
                      prediction: np.array,
                      ground_truth: np.array,
                      betas: list,
                      top_ks: list):
        """
        compute F-beta score

        input:
            + prediction: unthresholded output of the model, np.array of shape [B, N] with B be the 
            batchsize and N is the number of classes
            + ground_truth: self explanatory, must have the same shape as prediction
            + betas: a list of betas
            + top_ks: list of top_ks to compute the f_score
        output:
            + f_scores: (1+beta**2) * precision*recall/(beta**2 * precision+recall)
        """

        assert len(top_ks) > 0, "please specify top_k"

        outputs = {}

        for top_k in top_ks:
            # compute threshold for every top_k
            k_indices = np.argsort(prediction)[:, ::-1][:, top_k - 1]

            k_thresh = prediction[[range(len(k_indices)), k_indices]]
            k_thresh = k_thresh[..., np.newaxis]
            outputs["top_{}".format(top_k)] = self.f_k_score(
                prediction, ground_truth, betas, k_thresh)

        return outputs
