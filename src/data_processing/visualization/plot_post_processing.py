import matplotlib.pyplot as plt
line_args = {"linestyle": "--", "color": "k"}
plt.rcParams["figure.figsize"] = (20,5)

from cup_scripts.metric import fscore_step_detection
from cup_scripts.metric import fscore_step_detection_get_steps


def plot_post_processing(y0_pred, y0_pred_proc, y0_true, x0 = None):
    
    fig, axs = plt.subplots(3)
    fig.suptitle('Vertically stacked subplots')

    ys = [y0_pred, y0_pred_proc, y0_true]

    f1a, _ , _ = fscore_step_detection([y0_true],[y0_pred])
    f1b, _ , _ = fscore_step_detection([y0_true],[y0_pred_proc])

    pre, rec = fscore_step_detection_get_steps([y0_true],[y0_pred]).values()
    pre_true, pre_false = pre
    rec_true, rec_false = rec

    pre_p, rec_p = fscore_step_detection_get_steps([y0_true],[y0_pred_proc]).values()
    pre_true_p, pre_false_p = pre_p
    rec_true_p, rec_false_p = rec_p

    errors_pre = [pre_false[0], pre_false_p[0],[]]
    errors_rec = [rec_false[0], rec_false_p[0],[]]
        
    titles = [f"Prediction: {round(f1a,ndigits=4)}", f"Processed Prediction: {round(f1b,ndigits=4)}", "Ground truth"]
    colors = ["y", "orange", "g"]
    for i in range(3):
        
        error_pre = errors_pre[i]
        for (start,end) in error_pre:
            axs[i].axvline(start, **line_args)
            axs[i].axvline(end, **line_args)
            axs[i].axvspan(start, end, facecolor='r', alpha=0.5, hatch='/')

        error_rec = errors_rec[i]
        for (start,end) in error_rec:
            axs[i].axvline(start, **line_args)
            axs[i].axvline(end, **line_args)
            axs[i].axvspan(start, end, facecolor='r', alpha=0.5, hatch='\\')

        for (start,end) in ys[i]:
            axs[i].set_title(titles[i])
            axs[i].axvline(start, **line_args)
            axs[i].axvline(end, **line_args)
            axs[i].axvspan(start, end, facecolor=colors[i], alpha=0.5)

        if x0 is not None:
            sum = np.sum(x0,axis=1)
            axs[i].plot(sum)

    plt.tight_layout()
    plt.show()