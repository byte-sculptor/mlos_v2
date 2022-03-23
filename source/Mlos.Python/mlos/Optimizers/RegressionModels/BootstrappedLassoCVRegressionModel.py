
import logging
from mlos.Optimizers.RegressionModels.RegressionModel import RegressionModel



class BootstrappedLassoCVRegressionModel(RegressionModel):
    """Uses an enslemble of bootstrapped LassoCrossValidatedRegressionModels.

    This class may be very similar to HomogeneousRandomForestRegressionModel, so it would be good to
    abstract common patterns.

    """

    _PREDICTOR_OUTPUT_COLUMNS = [
        Prediction.LegalColumnNames.IS_VALID_INPUT,
        Prediction.LegalColumnNames.PREDICTED_VALUE,
        Prediction.LegalColumnNames.PREDICTED_VALUE_VARIANCE,
        Prediction.LegalColumnNames.SAMPLE_VARIANCE,
        Prediction.LegalColumnNames.SAMPLE_SIZE,
        Prediction.LegalColumnNames.PREDICTED_VALUE_DEGREES_OF_FREEDOM
    ]


    @trace()
    def __init__(
        self,
        model_config: Point,
        input_space: Hypergrid,
        output_space: Hypergrid,
        logger=None
    ):
        if logger is None:
            logger = create_logger(self.__class__.__name__)
        self.logger = logger

        assert model_config in bootstrapped_lasso_cv_config_store.parameter_space

        RegressionModel.__init__(
            self,
            model_type=type(self),
            model_config=model_config,
            input_space=input_space,
            output_space=output_space
        )
