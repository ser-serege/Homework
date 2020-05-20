u = train.groupBy('user_id').sum('purchase').select('user_id', f.col('sum(purchase)').alias('sum_user'))
t = train.groupBy('item_id').sum('purchase').select('item_id', f.col('sum(purchase)').alias('sum_item'))

train2 = train .join(u, on='user_id', how='left')\
               .join(t, on='item_id', how='left')

train2 = train2.fillna(0)

train2 = train2.withColumn('targ', (f.col('sum_user') + f.col('sum_item')) / 2)


#0.910123208718 rank=15, maxIter=10, regParam=0.05, alpha=1.0

als = ALS(coldStartStrategy="nan",  rank=15, maxIter=10, regParam=0.05, alpha=1.0, \
          userCol='user_id', itemCol='item_id', ratingCol='purchase', \
          nonnegative=False, implicitPrefs=True, seed=777)
als_model = als.fit(train2)

predict_test = als_model.transform(test)

@f.pandas_udf(FloatType())
def to_probs(values):
    return values.apply(lambda x: (x / 1.02))

pred = predict_test.withColumn("purchase", to_probs(f.col("prediction")))
pred = pred.sort(['user_id', 'item_id'], ascending=[True, True])

pred.select('user_id', 'item_id', 'purchase').toPandas().to_csv('lab03.csv', index=False)
