import matplotlib.pyplot as plt
import numpy as np

import external_pendulum as ep
import openai_pendulum as op


total_time_ep = []
total_time_op = []

average_time_ep = []
average_time_op = []

for episodes in [100, 200]:
    for iterations in [1000, 2000]:
        total, ep_average = ep.run(iterations, episodes)
        total_time_ep.append(total)
        total, op_average = op.run(iterations, episodes)
        total_time_op.append(total)

    average_time_ep.append(ep_average)
    average_time_op.append(op_average)

count = np.arange(4)
av_count = np.arange(2)
width = 0.35

fig1, ax1 = plt.subplots()

ep_bar = ax1.bar(count, total_time_ep, width)
op_bar = ax1.bar(count + width, total_time_op, width)

ax1.set_ylabel('Time (s)')
ax1.set_title('Total time by iterations and episodes')
ax1.set_xticks(count + width / 2)
ax1.set_xticklabels(('1000, 100', '1000, 200', '2000, 100', '2000, 200'))

ax1.legend((ep_bar[0], op_bar[0]), ('Without Gym', 'With Gym'))


fig2, ax2 = plt.subplots()

av_ep_bar = ax2.bar(av_count, average_time_ep, width)
av_op_bar = ax2.bar(av_count + width, average_time_op, width)

ax2.set_ylabel('Time (s)')
ax2.set_title('Average time by episodes')
ax2.set_xticks(av_count + width / 2)
ax2.set_xticklabels(('100', '200'))

ax2.legend((av_ep_bar[0], av_op_bar[0]), ('Without Gym', 'With Gym'))

plt.show()

