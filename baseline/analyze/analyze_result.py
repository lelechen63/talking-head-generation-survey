import sys
sys.path.insert(0, '/home/cxu-serve/u1/gcui2/code/few-shot-vid2vid')

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import os
from tqdm import tqdm

from analyze.util import oned_smooth
from util.util import rt_to_degree

import pdb

# get video key for specific dataset
def get_video_key(method, dataset, video):
    if 'yousaidthat' not in method and 'atvg' not in method:
        if dataset=='vox':
            video_key = video.replace('/', '_')+'_aligned'
        elif dataset == 'lrs':
            video_key = 'test_'+video.replace('/', '_')+'_crop'
        elif dataset == 'lrw':
            video_key = '_'.join(video.split('/')[-3:])+'_crop'
    elif dataset == 'vox' or dataset == 'lrs':
        video_key = video.replace('/', '_')
    elif dataset == 'lrw':
        video_key = '_'.join(video.split('/')[-3:])

    return video_key

# group degree to bins and save scores
def degree_result_from_csim(degree_pickle, video_pickle, csim_npy, save_pickle, method, dataset='vox'):
    # get bin
    with open(video_pickle, 'rb') as f:
        video = pickle.load(f)
        bin_list = video['degree']
        video = {}
    # get degree
    with open(degree_pickle, 'rb') as f:
        degree_dict = pickle.load(f)
    # get csim
    csim_dict = np.load(csim_npy).item()

    # separate result for each frame
    # get bins
    check_video = 0
    shift = max(bin_list)
    bin_len = bin_list[1] - bin_list[0]
    bin_dict = {bin_id:[] for bin_id in range(len(bin_list)-1)}
    for degrees, video in zip(degree_dict['degree'], degree_dict['img']):
        # count
        video_key = get_video_key(method, dataset, video)
        if video_key not in csim_dict:
            continue
        check_video += 1
        for d_i, d in enumerate(degrees[:, 1]+shift):
            if d > shift*2 or d < 0:
                continue
            bin_dict[d//bin_len].append([video, d_i])

    print('check video:{}'.format(check_video))
    # get min frame
    min_frame = min([len(bin_dict[bin_id]) for bin_id in bin_dict])
    print('min_frame:{}'.format(min_frame))

    # set bins result
    bin_result = {bin_id:[] for bin_id in bin_dict}
    for bin_id in bin_result:
        bin_datas = np.asarray(bin_dict[bin_id])
        for bin_data in bin_datas:
            video_key = get_video_key(method, dataset, bin_data[0])
            frame_id = int(bin_data[1])
            if frame_id >= csim_dict[video_key].shape[0] or csim_dict[video_key][frame_id] is None:
                continue
            bin_result[bin_id].append(csim_dict[video_key][frame_id])

    # store
    bin_result = [bin_result, bin_list]
    with open(save_pickle, 'wb') as f:
        pickle.dump(bin_result, f, protocol=pickle.HIGHEST_PROTOCOL)

    return bin_result

# group degree to motion and save scores
def motion_result_from_csim(motion_pickle, origin_npy, method, dataset='vox'):
    # get motion
    with open(motion_pickle, 'rb') as f:
        motions = pickle.load(f)
        total_dict = motions[0]
        motion_list = motions[2]
        motions = None
    # get csim
    total_score = np.load(origin_npy).item()

    # set bins result
    bin_result = {bin_id:[] for bin_id in total_dict}

    # get score
    for bin_id in bin_result:
        videos = total_dict[bin_id]
        for video in videos:
            video_key = get_video_key(method, dataset, video)
            try:
                score_list = [score for score in total_score[video_key] if score is not None]
            except:
                pdb.set_trace()
            bin_result[bin_id].append(np.mean(score_list))

    # clean
    video_num = min([len(bin_result[bin_id]) for bin_id in bin_result])
    print('min video number is {}'.format(video_num))

    return [bin_result, motion_list]

# group degree for calculating matrix
def matrix_result_from_score(matrix_pickle, score_npy, save_pickle, method, dataset='vox'):
    # load matrix
    with open(matrix_pickle, 'rb') as f:
        matrix_dict = pickle.load(f)
    # load score
    score_dict = np.load(score_npy).item()
    for video in score_dict:
        score_dict[video] = score_dict[video].item()

    # define matrix result
    bin_list = matrix_dict.pop('bin_list')
    m_keys = list(matrix_dict.keys())
    matrix_temp = {m_bin:{m_bin_2:[] for m_bin_2 in m_keys} for m_bin in m_keys}
    matrix_result = np.zeros((len(m_keys), len(m_keys)))
    
    # score to matrix
    video_store = []
    for m_bin in tqdm(m_keys):
        for m_bin_2 in tqdm(matrix_dict[m_bin]):
            video_list = matrix_dict[m_bin][m_bin_2]
            for video in video_list:
                video_store.append(video[0])
                video_key = get_video_key(method, dataset, video[0]) + '_{}'.format(video[1])
                score = score_dict[video_key]['%05d'%(video[2])]
                if score is not None:
                    matrix_temp[m_bin][m_bin_2].append(score)
    
    # get matrix result
    min_frame = None
    for m_bin in m_keys:
        for m_bin_2 in m_keys:
            matrix_result[m_bin][m_bin_2] = np.mean(matrix_temp[m_bin][m_bin_2])
            if min_frame is None or min_frame > len(matrix_temp[m_bin][m_bin_2]):
                min_frame = len(matrix_temp[m_bin][m_bin_2])

    # print info
    print('\nfor {} {}'.format(dataset, method))
    print('min_frame:{}'.format(min_frame))
    print('video_num:{}'.format(len(list(set(video_store)))))

    return bin_list, matrix_result

# plot results as histogram
def plot_result(bin_result, fig_path):
    bin_list = bin_result[1]
    bin_result = bin_result[0]
    bin_edge = ['[{}, {}]'.format(bin_list[i], bin_list[i+1]) for i in range(len(bin_list)-1)]
    hist = [np.mean(bin_result[bin_value]) for bin_value in bin_result]
    fig=plt.figure()
    plt.bar(bin_edge, hist, width=0.5)
    plt.xticks(rotation=70)
    fig.tight_layout()
    plt.savefig(fig_path)

# results as boxplot
def boxplot_result(bin_result, fig_path):
    bin_list = bin_result[1]
    bin_result = bin_result[0]
    bin_edge = ['[{}, {}]'.format(bin_list[i], bin_list[i+1]) for i in range(len(bin_list)-1)]
    hist = [bin_result[bin_value] for bin_value in bin_result]
    positions = np.arange(len(hist)) * 2
    fig, ax=plt.subplots()
    ax.boxplot(hist, positions=positions, showfliers=False)
    plt.xticks(positions, bin_edge, rotation=70)
    fig.tight_layout()
    plt.savefig(fig_path)

# plot several methods in one histogram
def plot_overall(result_dict, evaluations, fig_path):
    methods = list(result_dict.keys())
    colors = ['r', 'k', 'g', 'b', 'y', 'c']
    index = np.arange(len(methods))*len(evaluations)*1.5
    barwidth = 1
    plt.figure()
    for ev_id, evaluation in enumerate(evaluations):
        cur_index = index + barwidth * ev_id
        hist = [result_dict[method][ev_id] if len(result_dict[method]) > ev_id else 0 for method in result_dict]
        plt.bar(cur_index, hist, width=barwidth, align='edge', color=colors[ev_id])
    tick_index = index + barwidth * (len(evaluations)//2)
    plt.xticks(tick_index, methods, rotation= 70)

    plt.tight_layout()
    plt.savefig(fig_path)

# print confusion matrix
def confusion_matrix(matrix_result, fig_path, bin_list):
    # dataframe
    matrix_df = pd.DataFrame(matrix_result, \
                            index=bin_list, \
                            columns=bin_list)

    # set seaborn
    fig = plt.figure()
    sn.heatmap(matrix_df, cmap='YlGnBu', vmin=0, vmax=450)
    fig.tight_layout()
    plt.xticks(rotation=70)
    plt.savefig(fig_path)

# print boxplot for scores of several methods
def boxplot_several_result(bin_results, methods, fig_path):
    # preprocess
    bin_list = bin_results[0][1]
    for result_id in range(len(bin_results)):
        bin_results[result_id] = bin_results[result_id][0]
    # draw histogram
    barwidth = np.asarray([0.35 for i in range(len(bin_list)-1)])
    bin_edge = ['[{}, {}]'.format(round(bin_list[i],1), round(bin_list[i+1],1)) for i in range(len(bin_list)-1)]
    colors = ['r', 'k', 'g', 'b', 'y', 'c']
    indexes = np.arange(len(bin_list)-1) * (len(bin_results)) * 1.5
    boxes = []
    fig, ax = plt.subplots(figsize=(6.4*9,4.8*3))
    for i in range(len(bin_results)):
        cur_index = indexes + i * barwidth * 3
        hist = [bin_results[i][bin_value] for bin_value in bin_results[i]]
        b = ax.boxplot(hist, positions=cur_index, widths=barwidth+0.2, showfliers=False, \
            patch_artist=True, showmeans=True)
        # plt.setp(b['boxes'], color=colors[i])
        for patch in b['boxes']:
            patch.set_facecolor(colors[i])
        plt.setp(b['whiskers'], color=colors[i], linewidth=3)
        plt.setp(b['medians'], linewidth=4, color = 'c')
        plt.setp(b['means'], marker='^', markersize=8, mfc='w', mec='w', mew=2)
        boxes.append(b['boxes'][0])
    # other setting
    tick_index = indexes + barwidth * (len(bin_results)//2)
    plt.xticks(tick_index, bin_edge, rotation=70, fontsize=30)
    plt.yticks(fontsize=30)
    
    plt.ylim(0, 550)
    plt.xlim(-1, indexes[-1]+len(bin_results)*barwidth[0]*3)
    plt.tight_layout()
    plt.savefig(fig_path)

# group degree into bin and plot bin distribution
def get_each_figure():
    method = 'atvg'
    degree_pickle = 'analyze/degree_store/vox/test_degree.pkl'
    video_pickle = 'analyze/degree_store/vox/video_sel.pkl'
    csim_npy = 'degree/vox_results/csim/{}.npy'.format(method)
    save_pickle = 'analyze/degree_store/vox/bin_csim_result_{}.pkl'.format(method)
    fig_path = 'analyze/degree_store/vox/figures/csim_plot_vox_{}.png'.format(method)

    bin_result = degree_result_from_csim(degree_pickle, video_pickle, csim_npy, save_pickle, method)
    boxplot_result(bin_result, fig_path)

# plot bin result of several methods for each dataset
def get_multi_figure():
    evaluation = 'fid'
    datasets = ['vox', 'lrs', 'lrw']
    methods = ['wang', 'baseline', 'x2face', 'iccv', 'atvg', 'yousaidthat']
    for dataset in datasets:
        print('evaluaton {} for dataset {}'.format(evaluation, dataset))
        bin_results = []
        degree_pickle = 'analyze/degree_store/{}/test_degree.pkl'.format(dataset)
        video_pickle = 'analyze/degree_store/{}/video_sel.pkl'.format(dataset)
        fig_path = 'analyze/degree_store/{}/figures/{}_plot_{}_all.png'\
                    .format(dataset, evaluation, dataset)
        for method in methods:
            csim_npy = 'degree/{}_results/{}/{}.npy'.format(dataset, evaluation, method)
            save_pickle = 'analyze/degree_store/{}/bin_{}_result_{}.pkl'\
                .format(dataset, evaluation, method)
            bin_result = degree_result_from_csim(degree_pickle, video_pickle, csim_npy, save_pickle, method, dataset=dataset)
            bin_results.append(bin_result)
        
        # plot
        boxplot_several_result(bin_results, methods, fig_path)

# plot smooth degree figure
def plot_degree():
    degree_pickle = '/home/cxu-serve/u1/gcui2/code/few-shot-vid2vid/analyze/degree_store/vox/test_degree.pkl'
    with open(degree_pickle, 'rb') as f:
        degree_dict = pickle.load(f)
    degree_fig = '/home/cxu-serve/u1/gcui2/code/few-shot-vid2vid/analyze/degree_store/vox/figures/example_smooth_{}.png'
    video_name = 'id07868/YpUx8PfOELg/00249'
    video_id = np.where(np.asarray(degree_dict['img'])==video_name)[0][0]
    for video_name, degree in zip(degree_dict['img'][video_id:video_id+1], degree_dict['degree'][video_id:video_id+1]):
        degree = degree[:, 1]
        degree = oned_smooth(degree, window_len=10)
        plt.figure()
        plt.plot(np.arange(len(degree)), degree)
        plt.savefig(degree_fig.format(video_name.replace('/', '_')))
        plt.close()

# plot degree figure for multi-methods
def plot_degrees():
    root = '/home/cxu-serve/p1/common/other/obama_fake'
    methods = ['baseline', 'iccv', 'wang', 'x2face', 'gt']
    colors = ['k', 'b', 'r', 'g', 'c']
    names = ['baseline', 'Zakharov et al. (2019)​', 'Wang et al. (2019)​', 'x2face', 'ground truth']
    degree_fig = '/home/cxu-serve/u1/gcui2/code/few-shot-vid2vid/analyze/degree_store/obama/figures'
    if not os.path.exists(degree_fig):
        os.makedirs(degree_fig)
    degree_fig = os.path.join(degree_fig, 'degrees_plot.png')
    # plot
    plt.figure()
    for m_id, method in enumerate(methods):
        print('current {}'.format(method))
        rt_file = os.path.join(root, method, 'test_crop_rt.npy')
        rt = np.load(rt_file)
        degrees = []
        for rt_i in range(rt.shape[0]):
            degrees.append(rt_to_degree(rt[rt_i:rt_i+1]))
        degrees = np.concatenate(degrees, axis=0)
        plt.plot(list(range(degrees.shape[0])), degrees[:, 1], color=colors[m_id], label=names[m_id])
    plt.legend()
    plt.savefig(degree_fig)

# plot scores for motion evaluatoin
def plot_multi_motion_figure():
    evaluations = ['csim', 'ssim', 'cpbd', 'fid']
    datasets = ['vox', 'lrs', 'lrw']
    methods = ['wang', 'baseline', 'x2face', 'iccv']

    # get results for each motion
    for dataset in datasets:
        for evaluation in evaluations:
            print('evaluation method {} for dataset {}'.format(evaluation, dataset))
            bin_results = []
            motion_pickle = 'analyze/move_store/{}/video_sel.pkl'.format(dataset)
            fig_path = 'analyze/motion_store/{}/figures'.format(dataset)
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)
            for method in methods:
                origin_npy = 'degree/{}_results/{}/{}.npy'.format(dataset, evaluation, method)
                bin_result = motion_result_from_csim(motion_pickle, origin_npy, method, dataset=dataset)
                bin_results.append(bin_result)

            # plot
            boxplot_several_result(bin_results, methods, \
                os.path.join(fig_path, '{}_plot_{}_all.png'.format(evaluation, dataset)))

# print confusion matrix of several methods for each dataset
def plot_conf_matrix():
    datasets = ['vox', 'lrs', 'lrw']
    methods = ['wang', 'baseline', 'iccv']
    evaluation = 'fid'
    for dataset in datasets:
        for method in methods:
            matrix_pickle = 'analyze/degree_store/{}/matrix_sel.pkl'\
                .format(dataset)
            score_npy = 'matrix_result/{}_result/{}/{}.npy'.format(dataset, evaluation, method)
            save_pickle = ''
            fig_path = 'analyze/degree_store/{}/matrix'.format(dataset)
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)
            # get result
            bin_list, matrix_result = matrix_result_from_score(matrix_pickle, score_npy, save_pickle, method, dataset)
            bin_list = bin_list - max(bin_list) // 2
            
            index = ['[{},{})'.format(round(bin_list[bin_id],1), round(bin_list[bin_id+1],1)) for bin_id in range(len(bin_list)-1)]
            confusion_matrix(matrix_result, \
                            os.path.join(fig_path, '{}_{}_{}_matrix.png'.format(dataset, method, evaluation)), \
                            index)

# plot average scores of several method for each dataset
def plot_overall_exec():
    evaluations = ['csim', 'ssim', 'cpbd', 'fid']
    datasets = ['vox', 'lrs', 'lrw']
    methods = ['wang', 'baseline', 'x2face', 'iccv', 'yousaidthat', 'atvg']
    for dataset in datasets:
        result_dict = {method:[] for method in methods}
        result_dict.update({method:[] for method in other_methods})
        for evaluation in evaluations:
            print('evaluation method {} for dataset {}'.format(evaluation, dataset))
            motion_pickle = '/home/cxu-serve/u1/gcui2/code/few-shot-vid2vid/analyze/move_store/{}/video_sel.pkl'.format(dataset)
            for method in methods:
                origin_npy = '/home/cxu-serve/p1/common/degree/{}_results/{}/{}.npy'.format(dataset, evaluation, method)
                extra_npy = '/home/cxu-serve/p1/common/motion/{}_results/{}/{}.npy'.format(dataset, evaluation, method)
                if not os.path.exists(origin_npy):
                    print('{} not exist'.format(origin_npy))
                    continue
                bin_result = motion_result_from_csim(motion_pickle, extra_npy, origin_npy, method, dataset=dataset)
                result_dict[method].append(np.mean(list(bin_result[0].values())))

        # plot
        fig_path = '/home/cxu-serve/u1/gcui2/code/few-shot-vid2vid/analyze/overall/{}/figures'.format(dataset)
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        fig_path = os.path.join(fig_path, "plot_{}_all.png".format(dataset))
        plot_overall(result_dict, evaluations, fig_path)

# plot histogram for accuracy
def plot_accuracy():
    methods = ['atvg', 'yousaidthat', 'ijcv', 'baseline', 'iccv', 'x2face', 'wang', 'gt']
    colors = ['pink', 'c', 'm', 'gray', 'green', 'xkcd:sky blue', 'orange', 'olive']
    sections = ['overall', 'blink', 'not blink']
    overall = [0.5273, 0.4824, 0.5195, 0.7695, 0.5391, 0.7188, 0.6875, 0.7871]
    blinks = [0.544, 0.443, 0.4691, 0.7655, 0.6026, 0.7492, 0.7394, 0.7622]
    unblinks = [0.5024, 0.5415, 0.5951, 0.7756, 0.4439, 0.6732, 0.6098, 0.8244]

    plt.plot()
    indexs = np.arange(3)*(len(overall)+2)
    barwidth = 1
    for method_id in range(len(methods)):
        hist = [overall[method_id], blinks[method_id], unblinks[method_id]]
        plt.bar(indexs+method_id, hist, barwidth, color=colors[method_id], align='edge')
    tick_index = indexs + len(overall) // 2
    plt.xticks(tick_index, sections)

    plt.tight_layout()
    plt.savefig('accuracy.png')
    
if __name__ == '__main__':
    # get_multi_figure()
    # plot_degree()
    # plot_results_degree()
    # get_each_figure()
    # plot_conf_matrix()
    # plot_multi_motion_figure()
    # plot_overall_exec()
    # plot_degrees()
    plot_accuracy()