class CorrelattionsRemover(BaseEstimator, TransformerMixin):
    
    def __init__(self, args):
        self.features_dataframe, yam = args
        self.target_name = target_name(yam)
        self.feature_cols = feature_cols(yam)
        self.corr_treshold = corr_treshold(yam)
        self.corr_ = None
        self.corr_cols = None
        self.corr_pairs = None
        self.corr_matrix = None
        self.corr_with_target = None
        
    def fit(self, args, y=None):
        if type(dfs) == dict:
            x_train, y_train = unpack_df(dfs, what=['x_train'])
        else:
            x_train, y_train = dfs
            n=200000
            if len(dfs) > n:
                x_train = dfs.sample(n=n, random_state=77)
            else:
                x_train = dfs.copy()
            self.corr_with_target = abs(x_train.corr())[self.target_name]
            
        if self.feature_cols is None:    
            self.feature_cols = x_train.columns.tolist()
            
        self.feature_cols = [col for col in self.feature_cols if col in x_train.columns and col !=self.target_name]
        df2 = x_train.fillna(0)
        self.corr_matrix = np.corrcoef(df2[self.feature_cols].values, rowvar=False)
        big_corr_ind = np.where(abs(self.corr_matrix) >= self.corr_treshold)
        big_corr_pairs = [(self.feature_cols[i], self.feature_cols[j]) for i, j
                          in zip(big_corr_ind[0], big_corr_ind[1]) if i > j]
        self.corr_cols = list({pair[1] for pair in self.corr_pairs})
        
        return self
    
    
    def transform(self, df, y=None):
        self.feature_dataframe['drop_reason'] = np.where(self.feature_dataframe['feature'].isin(self.corr_cols),
                                                         'Удалено кореллирующих признаков',
                                                         self.feature_dataframe['drop_reason']
                                                        )
        if type(df) == dict:
            if len(self.corr_cols) > 0:
                for dataset in df.keys():
                    df[dataset][0] = df[dataset][0].drop(self.corr_cols, axis=1)
            else:
                return df
        else:
            df = df.drop(columns=self.corr_cols)
        
        return df
    
    def fit_transform(self, df, y=None, **kwargs):
        self.fit(df)
        if len(self.drop_features) > 0:
            return self.corr_cols(dfs)
        

