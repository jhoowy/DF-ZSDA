import os
import time


class Logger(object):
    def __init__(self, cfg):
        self.log_name = os.path.join(cfg.checkpoints_dir, cfg.name, 'train_log.txt')
        with open(self.log_name, "w") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)
    
    def log_message(self, message):
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def print_evaluation_result(self, name, accuracy, di_accuracy=None, ens_accuracy=None):
        message = '======= Evaluation on %s =======\n' % name
        if di_accuracy == None:
            message += 'Evaluation: acc %.2f%%' % accuracy
        elif ens_accuracy == None:
            message += 'Evaluation: acc %.2f%%, di acc %.2f%%' % (accuracy, di_accuracy)
        else:
            message += 'Evaluation: acc %.2f%%, di acc %.2f%%, ens acc %.2f%%' % (accuracy, di_accuracy, ens_accuracy)
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def print_end_of_training(self, accuracy, real_best_accuracy=None):
        msg_width = len('######### End of training, rt-accuracy : %.2f #########' % (accuracy))
        message = '\n' + '#' * msg_width
        message += '\n######### End of training, rt-accuracy : %.2f #########' % (accuracy)
        message += '\n' + '#' * msg_width

        if real_best_accuracy != None:
            msg_width = len('######### Real best rt-accuracy : %.2f #########' % (real_best_accuracy))
            message += '\n\n' + '#' * msg_width
            message += '\n######### Real best rt-accuracy : %.2f #########' % (real_best_accuracy)
            message += '\n' + '#' * msg_width

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n\n' % message)

    def print_test_results(self, domain, accuracy, di_accuracy, ens_accuracy=None):
        message = '======= Test on %s domain =======\n' % domain
        if ens_accuracy == None:
            message += 'Test: acc %.2f%%, di acc %.2f%%' % (accuracy, di_accuracy)
        else:
            message += 'Test: acc %.2f%%, di acc %.2f%%, ens acc %.2f%%' % (accuracy, di_accuracy, ens_accuracy)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)