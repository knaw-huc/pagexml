from collections import Counter
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid")


def plot_dist_stats(stats: Dict[str, Dict[str, Counter]]) -> None:
    for doc_type in stats.keys():
        num_plots = len(stats[doc_type].keys())
        fig, axes = plt.subplots(1, num_plots, figsize=(16, 8))
        fig.suptitle(f'{doc_type.title()}')
        for fi, field in enumerate(stats[doc_type]):
            if field == "avg":
                continue
            min_val = min(stats[doc_type][field])
            max_val = max(stats[doc_type][field])
            spread = max_val - min_val
            factor = np.floor((np.log(spread) / np.log(10)))
            binwidth = np.round((np.log(spread) / np.log(10)) ** factor)
            points = [val for key, value in stats[doc_type][field].items() for val in [key] * value]
            df = pd.DataFrame(data={field: points})
            x, y = zip(*sorted(stats[doc_type][field].items()))
            sns.histplot(df, ax=axes[fi], x=field, binwidth=binwidth)
            # sns.kdeplot(ax=axes[fi], x=x, y=y, cut=0)
            axes[fi].set_title(field)
