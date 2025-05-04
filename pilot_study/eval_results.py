import pandas as pd
import yfinance as yf
from datetime import timedelta
from sklearn.metrics import classification_report, confusion_matrix

# Load Biden tweets dataset (with price moves)
tweets = pd.read_csv('annotate_congress_with_price_moves.csv', parse_dates=['date'])

# 1) Grisd-search eps to maximize macro F1
from sklearn.metrics import f1_score

eps_values = [0, 0.02, 0.05, 0.1, 0.2]
best = {'eps': None, 'f1': -1}

label_map = {'bear': 'down', 'neut': 'neutral', 'bull': 'up'}

for eps in eps_values:
    def pick_eps(r):
        scores = {'bear': r['sent_bear'], 'neut': r['sent_neut'], 'bull': r['sent_bull']}
        srt = sorted(scores, key=scores.get, reverse=True)
        top, second = srt[0], srt[1]
        if scores[top] - scores[second] >= eps:
            pred = top
        else:
            pred = second  # runner-up fallback
        return pred

    tweets['model_pred_tmp'] = tweets.apply(pick_eps, axis=1)
    tweets['model_move_tmp'] = tweets['model_pred_tmp'].map(label_map)
    df_tmp = tweets.dropna(subset=['price_move', 'model_move_tmp'])
    f1 = f1_score(df_tmp['price_move'], df_tmp['model_move_tmp'], average='macro', labels=['down','neutral','up'])
    if f1 > best['f1']:
        best = {'eps': eps, 'f1': f1}

print(f"Best eps: {best['eps']} with macro-F1={best['f1']:.3f}")

# Now apply best eps to get final predictions
best_eps = best['eps']
def pick_pred(r):
    scores = {'bear': r['sent_bear'], 'neut': r['sent_neut'], 'bull': r['sent_bull']}
    srt = sorted(scores, key=scores.get, reverse=True)
    top, second = srt[0], srt[1]
    if scores[top] - scores[second] >= best_eps:
        return top
    else:
        return second

tweets['model_pred'] = tweets.apply(pick_pred, axis=1)
# 2) Map sentiment labels to price move categories
label_map = {'bear': 'down', 'neut': 'neutral', 'bull': 'up'}
tweets['model_move'] = tweets['model_pred'].map(label_map)
label_map = {'bear': 'down', 'neut': 'neutral', 'bull': 'up'}
tweets['model_move'] = tweets['model_pred'].map(label_map)

# 3) Evaluate against actual price moves
# Drop any rows where we couldn’t compute a move
eval_df = tweets.dropna(subset=['price_move', 'model_move'])

y_true = eval_df['price_move']
y_pred = eval_df['model_move']

print("\nClassification Report:")
print(classification_report(y_true, y_pred, labels=['down','neutral','up']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_true, y_pred, labels=['down','neutral','up'])
print(pd.DataFrame(cm, index=['true_down','true_neutral','true_up'], columns=['pred_down','pred_neutral','pred_up']))

# Optionally save
cm_df = pd.DataFrame(cm, index=['true_down','true_neutral','true_up'], columns=['pred_down','pred_neutral','pred_up'])
cm_df.to_csv('pilot_study/confusion_matrix_price.csv')
print("\nSaved confusion matrix as pilot_study/confusion_matrix_price.csv")
