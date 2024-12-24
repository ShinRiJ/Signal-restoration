class MLP:
    def __init__(self, model=None, n_preds=None, n_regrs=None, pred_idxes=None, regr_idxes=None, lr=1e-2):
        # Проверка аргументов
        assert n_preds or pred_idxes
        assert n_regrs or regr_idxes

        # 
        self._n_preds = n_preds if n_preds else len(pred_idxes)
        self._n_regrs = n_regrs if n_regrs else len(regr_idxes)
        
        self._pred_idxes = pred_idxes
        self._regr_idxes = regr_idxes

        self._model = model if model else mlp_model(n_preds, n_regrs)
        self._model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)) #learning_rate=1e-02
        self._early_stopper = EarlyStopping('val_loss', min_delta=0.001, patience=3, verbose=1)
        self._lr_reducer = ReduceLROnPlateau('val_loss', factor=0.2, patience=2, min_lr=1e-6, verbose=1)            
        self._lr = lr
        self._init_epoch = 0
        self.history = None
        
        # StandardScaler


    def fit(self, train_data, valid_data, n_epochs, clear_model=False):
        if clear_model:
            tf.keras.backend.clear_session()   
            self._model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=self._lr, momentum=0.9))

        self.history = self._model.fit(
            train_data,
            validation_data=train_data,
            epochs=n_epochs,
            callbacks=[self._early_stopper, self._lr_reducer]
        )   


    def predict_window(self, data):
        result = data.copy()

        for i in tqdm(range(0, (len(data) - self._n_preds) // (self._n_regrs + 1))):
            X = data[self._pred_idxes + (self._n_preds + 1) * i]
            if (len(X.shape) == 1):
                X = data.reshape(1, -1)

            y_pred = self._model(X, verbose=0)
            result[self._rergr_idxes + (self._n_regrs + 1) * i] = y_pred

        return result
    