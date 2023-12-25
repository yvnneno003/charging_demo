#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#   Copyright (C) 2023 MSI-FUNTORO
#
#   Licensed under the MSI-FUNTORO License, Version 1.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       https://www.funtoro.com/global/
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
detect_evaluate_utils.py
"""
import os
import copy
import matplotlib.pyplot as plt
from colorama import init, Fore

init(autoreset=True)

class DetectEvaluateUtils:

    def calculate_iou(self, box1, box2):
        '''
        Calculate IoU

        Args:
            box1 (list): List of box1 location. [top x, top y, bottom x, bottom y]
            box2 (list): List of box2 location. [top x, top y, bottom x, bottom y]

        Returns:
            iou (float): Value of IoU.
        '''
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x1 >= x2 or y1 >= y2:
            return 0

        intersection = (x2 - x1) * (y2 - y1)
        union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - intersection
        iou = intersection / float(union)
        return iou


    def calculate_precision_recall(self, labels, predicts, targets, confidences_threshold=0.5, iou_threshold=0.7, show_plot=True, save_plot_path=None, show_message=True):
        '''
        Calculate precision and recall

        Args:
            labels (list): A list of labels.
            predicts (list): A list of predicts for each image. [[[label, box_top_x, box_top_y, box_bottom_x, box_bottom_y, confidence], [...]], [[...]]]
            targets (list): A list of targets for each image. [[[label, box_top_x, box_top_y, box_bottom_x, box_bottom_y], [...]], [[...]]]
            confidences_threshold (float): The confidence threshold.
            iou_threshold (float): The IoU threshold used for matching predicted and ground truth bounding boxes.
            show_plot (bool): Show plot for precision and recall.
            save_plot_path (bool): The path of save plot folder.
            show_message (bool): Show debug message.

        Returns:
            total_precision (float): Precision for total label.
            total_recall (float): Recall for total label.
            precisions_recalls (dictionary): Precision and recall for each label.
                    {'label': (str), 'precision': (float), 'recall': (float), 'tp': (float), 'tn': : (float), 'fp': (float), 'fn': (float)}
        '''
        _predicts = copy.deepcopy(predicts)
        _targets = copy.deepcopy(targets)

        #********** Total Precision Recall **********
        # get the predicts of all predict files, then sort by confidence
        predict_for_total_label = []
        for predict_file_index, file_predicts in enumerate(_predicts):
            for predict in file_predicts:
                predict.insert(0, predict_file_index)
                predict_for_total_label.append(predict)
        predict_for_total_label = sorted(predict_for_total_label, key=lambda x: -x[6])

        # get the targets of all target files
        targets_for_total_label = []
        for target_file_index, file_targets in enumerate(_targets):
            for target in file_targets:
                target.insert(0, target_file_index)
                target.append(0)
                targets_for_total_label.append(target)

        # calculate total precision and total recall
        total_precision, total_recall, _, _, _, _, ious = self._calculate_precision_recall(predict_for_total_label, targets_for_total_label, confidences_threshold, iou_threshold, show_message)
        if show_message:
            print('Total precision: ', total_precision)
            print('Total recall: ', total_recall)


        #********* The precision and recall for each label *********
        # get precision and recall for each labels
        precisions, recalls = [], []
        precisions_recalls = []
        for label_index, label in enumerate(labels):
            _predicts = copy.deepcopy(predicts)
            _targets = copy.deepcopy(targets)
            if show_message:
                print(Fore.BLUE + '\n' + label + '[' + str(label_index) + ']')

            # get the targets of all target files
            target_for_label = []
            for target_file_index, file_targets in enumerate(_targets):
                for target in file_targets:
                    if target[0] == label_index:
                        target.insert(0, target_file_index)
                        target.append(0)
                        target_for_label.append(target)
            if len(target_for_label) == 0:
                continue

            # get the predicts of all predict files, then sort by confidence
            predict_for_label = []
            for predict_file_index, file_predicts in enumerate(_predicts):
                for predict in file_predicts:
                    if predict[0] == label_index:
                        predict.insert(0, predict_file_index)
                        predict_for_label.append(predict)
            predict_for_label = sorted(predict_for_label, key=lambda x: -x[6])

            precision, recall, tp, tn, fp, fn, _ = self._calculate_precision_recall(predicts=predict_for_label, targets=target_for_label, confidences_threshold=confidences_threshold, iou_threshold=iou_threshold,
                                                                                    show_message=show_message)
            precisions.append({'label': label, 'precision': precision})
            recalls.append({'label': label, 'recall': recall})
            precisions_recalls.append({'label': label, 'precision': precision, 'recall': recall, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn})

            if show_message:
                print('Precision: ', precision)
                print('Recall: ', recall)
                print('TP: ' + str(tp) + ', TN: ' + str(tn) + ', FP: ' + str(fp) + ', FN: ' + str(fn))

        precisions_recalls = sorted(precisions_recalls, key=lambda x: x['precision'])
        self._draw_precision_plot(total_precision=total_precision, precisions=precisions_recalls, show=show_plot, save_path=save_plot_path)
        precisions_recalls = sorted(precisions_recalls, key=lambda x: x['recall'])
        self._draw_recall_plot(total_recall=total_recall, recalls=precisions_recalls, show=show_plot, save_path=save_plot_path)

        return total_precision, total_recall, precisions_recalls, ious


    def calculate_map(self, labels, predicts, targets, confidences_threshold=0.7, iou_threshold=0.5, show_plot=True, save_plot_path=None, show_message=True):
        '''
        Calculate mAP

        Args:
            labels (list): A list of labels.
            predicts (list): A list of predicts for each image. [[[label, box_top_x, box_top_y, box_bottom_x, box_bottom_y, confidence], [...]], [[...]]]
            targets (list): A list of targets for each image. [[[label, box_top_x, box_top_y, box_bottom_x, box_bottom_y], [...]], [[...]]]
            confidences_threshold (float): The confidence threshold.
            iou_threshold (float): The IoU threshold used for matching predicted and ground truth bounding boxes.
            show_plot (bool): Show plot for precision and recall.
            save_plot_path (bool): The path of save plot folder.
            show_message (bool): Show debug message.

        Returns:
            aps (dictionary): Average precision of each label.
                    {'label': (str), 'ap': (float)}
            mAP (float): mAP (mean average precision).
        '''
        aps = []

        for label_index, label in enumerate(labels):
            _predicts = copy.deepcopy(predicts)
            _targets = copy.deepcopy(targets)
            if show_message:
                print(Fore.BLUE + '\n' + label + '[' + str(label_index) + ']')

            # get the targets of all target files
            target_for_label = []
            for target_file_index, file_targets in enumerate(_targets):
                for target in file_targets:
                    if target[0] == label_index:
                        target.insert(0, target_file_index)
                        target.append(0)
                        target_for_label.append(target)
            if len(target_for_label) == 0:
                continue

            # get the predicts of all predict files, then sort by confidence
            predict_for_label = []
            for predict_file_index, file_predicts in enumerate(_predicts):
                for predict in file_predicts:
                    if predict[0] == label_index:
                        predict.insert(0, predict_file_index)
                        predict_for_label.append(predict)
            predict_for_label = sorted(predict_for_label, key=lambda x: -x[6])

            # calculate tp, tn, fp, fn
            tps, tns, fps, fns = [], [], [], []
            for predict in predict_for_label:
                _, _, tp, tn, fp, fn, _ = self._calculate_precision_recall(predicts=[predict], targets=target_for_label, confidences_threshold=confidences_threshold, iou_threshold=iou_threshold, show_message=show_message)
                tps.append(tp)
                tns.append(tn)
                fps.append(fp)
                fns.append(fn)

            _tps = copy.deepcopy(tps)
            cumsum = 0
            for idx, val in enumerate(_tps):
                _tps[idx] += cumsum
                cumsum += val
            _fps = copy.deepcopy(fps)
            cumsum = 0
            for idx, val in enumerate(_fps):
                _fps[idx] += cumsum
                cumsum += val

            if show_message:
                print('True positive: ', tps, '->', _tps)
                print('False positive: ', fps, '->', _fps)

            # calculate precisions and recalls
            _precisions = _tps[:]
            _recalls = _tps[:]
            for index, (tp, fp) in enumerate(zip(_tps, _fps)):
                _precisions[index] = float(tp) / (tp + fp)
                if len(target_for_label) != 0:
                    _recalls[index] = float(tp) / len(target_for_label)
                else:
                    _recalls[index] = 0

            if show_message:
                print('Precisions: ', _precisions)
                print('Recalls: ', _recalls)

            # calculate ap
            ap, ap_precisions, ap_recalls = self.calculate_ap(precisions=_precisions[:], recalls=_recalls[:])
            aps.append({'label': label, 'ap': ap})

            if show_message:
                print('AP: {:.2f}%'.format(ap * 100))

            # draw [ap, precision, recall] plot
            save_path = None if save_plot_path is None else os.path.join(save_plot_path, '[AP] ' + str(label) + '.jpg')
            self._draw_ap_precision_recall_plot(label=label, ap=ap, precisions=_precisions, recalls=_recalls, ap_precisions=ap_precisions, ap_recalls=ap_recalls, show=show_plot,
                                                save_path=save_path)

        # calculate mAP
        ap_sum = 0
        for ap in aps:
            ap_sum += ap['ap']
        if len(aps) == 0:
            mAP = 0
        else:
            mAP = ap_sum / len(aps)

        # sort ap list by ap
        aps = sorted(aps, key=lambda x: x['ap'])

        # draw mAP to plot
        self._draw_map_plot(ap_list=aps, mAP=mAP, show=show_plot, save_path=save_plot_path)

        return aps, mAP


    def calculate_ap(self, precisions, recalls):
        '''
        Calculate AP (Average Precision)

        Args:
            precisions (list): A list of precisions.
            recalls (list): A list of recalls.

        Returns:
            ap (float):
            _precisions (list):
            _recalls (list):
        '''
        recalls.insert(0, 0.0)
        recalls.append(1.0)
        _recalls = recalls[:]

        precisions.insert(0, 0.0)
        precisions.append(0.0)
        _precisions = precisions[:]

        for i in range(len(_precisions) - 2, -1, -1):
            _precisions[i] = max(_precisions[i], _precisions[i + 1])

        i_list = []
        for i in range(1, len(_recalls)):
            if _recalls[i] != _recalls[i - 1]:
                i_list.append(i)

        ap = 0.0
        for i in i_list:
            ap += ((_recalls[i] - _recalls[i - 1]) * _precisions[i])

        return ap, _precisions, _recalls


    def _calculate_precision_recall(self, predicts, targets, confidences_threshold=0.5, iou_threshold=0.7, show_message=True):
        # initialize tp, tn, fp, fn, precision, recall
        true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0
        precision, recall = 0, 0
        ious = []

        for predict in predicts:
            match_found = False
            match_index = 0
            max_iou = -1

            for target_index, target in enumerate(targets):
                # check used, file index, label index and confidence
                if target[6] == 1 or predict[0] != target[0] or predict[1] != target[1] or predict[6] < confidences_threshold:
                    continue

                # calculate IoU
                iou = self.calculate_iou(predict[2:6], target[2:6])

                if iou > max_iou:
                    max_iou = iou
                    if iou >= iou_threshold:
                        match_found = True
                        match_index = target_index
            ious.append(max_iou)

            if match_found:
                true_positive += 1
                targets[match_index][6] = 1
            else:
                false_positive += 1

            if show_message:
                if match_found:
                    color = Fore.GREEN
                else:
                    color = Fore.RED
                boxes_str = ', '.join(str(i) for i in predict[2:6])
                print(color + 'Label: ' + str(predict[1]) + ', Box: [' + boxes_str + '], Confidence: ' + str(predict[6]) + ', IoU: ' + str(max_iou))


        # calculate false negative
        false_negative = len(targets) - true_positive

        if show_message:
            print(Fore.GREEN + 'TP: ' + str(true_positive) + Fore.RED + ', TN: ' + str(true_negative))
            print(Fore.GREEN + 'FP: ' + str(false_positive) + Fore.RED + ', FN: ' + str(false_negative))

        if len(predicts) != 0 and len(targets) != 0:
            # calculate precision
            precision = true_positive / (true_positive + false_positive)

            # calculate true negative
            true_negative = len(predicts) - true_positive - false_positive

            # calculate recall
            recall = true_positive / (true_positive + false_negative)

        return precision, recall, true_positive, true_negative, false_positive, false_negative, ious


    def _draw_ap_precision_recall_plot(self, label, ap, precisions, recalls, ap_precisions, ap_recalls, show, save_path):
        plt.plot(recalls, precisions, '-o')

        # add a new penultimate point to the list (mrec[-2], 0.0)
        # since the last line segment (and respective area) do not affect the AP value
        area_under_curve_x = ap_recalls[:-1] + [ap_recalls[-2]] + [ap_recalls[-1]]
        area_under_curve_y = ap_precisions[:-1] + [0.0] + [ap_precisions[-1]]
        plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')

        # set window title
        fig = plt.gcf()  # gcf - get current figure
        # fig.canvas.set_window_title('AP ' + label)
        # set plot title
        plt.title('[' + label + '] AP: {:.2f}%'.format(ap * 100))
        plt.xlabel('Recall')
        plt.ylabel('Precision')

        # optional - set axes
        axes = plt.gca()  # gca - get current axes
        axes.set_xlim([0.0, 1.0])
        axes.set_ylim([0.0, 1.05])  # .05 to give some extra space

        if save_path is not None:
            fig.savefig(save_path)
        if show:
            plt.show()
        plt.cla()  # clear axes for next plot


    def _draw_map_plot(self, ap_list, mAP, show, save_path):
        labels = []
        aps = []
        for ap in ap_list:
            labels.append(ap['label'])
            aps.append(ap['ap'])

        _save_path = None
        if save_path is not None:
            _save_path = os.path.join(save_path, 'mAP.jpg')
        self._draw_plot(title='mAP = {:.2f} %'.format(mAP * 100),
                        labels=labels,
                        values=aps,
                        value_tag_str='Average Precision',
                        color='royalblue',
                        show=show,
                        save_path=_save_path)


    def _draw_precision_plot(self, total_precision, precisions, show, save_path):
        _labels, _precision = [], []
        for precision in precisions:
            _labels.append(precision['label'])
            _precision.append(precision['precision'])

        _save_path = None
        if save_path is not None:
            _save_path = os.path.join(save_path, 'precision.jpg')
        self._draw_plot(title='Precision = {:.2f} %'.format(total_precision * 100),
                        labels=_labels,
                        values=_precision,
                        value_tag_str='Precision',
                        show=show,
                        save_path=_save_path)


    def _draw_recall_plot(self, total_recall, recalls, show, save_path):
        _labels, _recall = [], []
        for recall in recalls:
            _labels.append(recall['label'])
            _recall.append(recall['recall'])

        _save_path = None
        if save_path is not None:
            _save_path = os.path.join(save_path, 'recall.jpg')
        self._draw_plot(title='Recall = {:.2f} %'.format(total_recall * 100),
                        labels=_labels,
                        values=_recall,
                        value_tag_str='Recall',
                        show=show,
                        save_path=_save_path)


    def _draw_plot(self, title, labels, values, value_tag_str, color='royalblue', show=True, save_path=None):
        # set title
        plt.title(title, fontsize=16)
        fig = plt.gcf()
        axes = plt.gca()

        # set values bar
        plt.barh(range(len(values)), values, color=color)

        # set labels to y tick
        plt.yticks(range(len(labels)), labels, fontsize=12)

        # set value bar string
        r = fig.canvas.get_renderer()
        for index, value in enumerate(values):
            value_str = ' ' + str(value * 100) + ' %'
            if value < 1.0:
                value_str = ' {:.2f} %'.format(value * 100)
            t = plt.text(value, index, value_str, color=color, va='center', fontweight='bold')
            if index == (len(values)-1): # largest bar
                self._adjust_axes(r, t, fig, axes)

        plt.xlabel(value_tag_str, fontsize=12)
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path)
        if show:
            plt.show()
        plt.cla()
        plt.close()



    def _adjust_axes(self, r, t, fig, axes):
        # get text width for re-scaling
        bb = t.get_window_extent(renderer=r)
        text_width_inches = bb.width / fig.dpi
        # get axis width in inches
        current_fig_width = fig.get_figwidth()
        new_fig_width = current_fig_width + text_width_inches
        propotion = new_fig_width / current_fig_width
        # get axis limit
        x_lim = axes.get_xlim()
        axes.set_xlim([x_lim[0], x_lim[1] * propotion])




if __name__ == '__main__':
    detect_evaluate_utils = DetectEvaluateUtils()

    labels = ['tv monitor', 'cup', 'book', 'bottle', 'chair', 'potted plant', 'picture frame', 'heater', 'coffee table', 'book case', 'doll', 'vase', 'waste container', 'nightstand', 'refrigerator']

    predicts = [[[2, 433, 260, 506, 336, 0.269833],
                 [2, 518, 314, 603, 369, 0.462608],
                 [2, 592, 310, 634, 388, 0.298196],
                 [2, 403, 384, 517, 461, 0.382881],
                 [2, 405, 429, 519, 470, 0.369369],
                 [2, 433, 272, 499, 341, 0.272826],
                 [2, 413, 390, 515, 459, 0.619459]],
                [[2, 529, 201, 593, 309, 0.9],
                 [12, 100, 100, 200, 200, 0.2],
                 [0, 100, 100, 300, 300, 0.2]]]

    targets = [[[2, 439, 157, 556, 241],
                [2, 437, 246, 518, 351],
                [2, 515, 306, 595, 375],
                [2, 407, 386, 531, 476],
                [2, 544, 419, 621, 476],
                [2, 609, 297, 636, 392]],
               [[2, 528, 213, 602, 300],
                [12, 100, 100, 200, 200],
                [0, 100, 200, 300, 300]]]

    total_precision, total_recall, precisions_recalls = detect_evaluate_utils.calculate_precision_recall(labels=labels,
                                                                                                         predicts=predicts,
                                                                                                         targets=targets,
                                                                                                         confidences_threshold=0.2,
                                                                                                         iou_threshold=0.5,
                                                                                                         show_plot=True,
                                                                                                         save_plot_path='../outputs',
                                                                                                         show_message=False)
    print('\n================================')
    print('Total precision:', total_precision)
    print('Total recall: ', total_recall)
    print('Precision and recall: ')
    for precision_recall in precisions_recalls:
        print('\t', precision_recall)
    print('================================')


    aps, mAP = detect_evaluate_utils.calculate_map(labels=labels,
                                                   predicts=predicts,
                                                   targets=targets,
                                                   confidences_threshold=0.2,
                                                   iou_threshold=0.5,
                                                   show_plot=True,
                                                   save_plot_path='../outputs',
                                                   show_message=False)
    print('\n================================')
    print('AP list: ')
    for ap in aps:
        print('\t', ap)
    print('mAP: ', mAP)
    print('================================')