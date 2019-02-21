#!/usr/bin/env python

"""Prints different output formats."""

import logging

import tensorflow as tf

import inference
import util
import json
from collections import OrderedDict


class OutputHandler(object):
    def __init__(self, output_file, output_format, vocab):
        """

        :param output_file:
        :param output_format:
        :param vocab:
        """
        self.output_file = output_file
        self.output_format = output_format
        self.vocab = vocab

        if self.output_format == "translation":
            self.write = self.print_translation
        elif self.output_format == "nbest":
            self.write = self.print_nbest
        elif self.output_format == "json":
            self.write = self.print_json
        elif self.output_format == "soft":
            self.write = self.print_soft
            
    def get_words(self, line, add_eos=True):
        """

        :param line:
        :param add_eos:
        :return:
        """
        words = line.split()
        if add_eos:
            words.append('<eos>')
        return words
        
    def print_translation(self, beams, alignments, maxibatch, num_prev_translated):
        """
        Prints only the 1-best translation itself.

        :param beams: n best translations each with their score, for a batch of inputs.
                      Shape: (batch, beam), individual beam items are (translation, score)
        :param alignments: soft attention scores between all source and target words.
                           Shape: (batch, num_models, beam, target_seq_len, source_seq_len)
        :param maxibatch: a list of input (source) sentences
        :param num_prev_translated: how many sentences were translated in this file before
                                    this specific batch
        """
        for beam in beams:
            best_hypo, cost = beam[0]
            line = util.seq2words(best_hypo, self.vocab) + '\n'
            self.output_file.write(line)


    def print_nbest(self, beams, alignments, maxibatch, num_prev_translated):
        """
        Prints n best translations for each source sentence in a batch.

        :param beams: n best translations each with their score, for a batch of inputs.
                      Shape: (batch, beam), individual beam items are (translation, score)
        :param alignments: soft attention scores between all source and target words.
                           Shape: (batch, num_models, beam, target_seq_len, source_seq_len)
        :param maxibatch: a list of input (source) sentences
        :param num_prev_translated: how many sentences were translated in this file before
                                    this specific maxibatch
        """
        for i, beam in enumerate(beams):
            num = num_prev_translated + i
            for sent, cost in beam:
                translation = util.seq2words(sent, self.vocab)
                line = "{} ||| {} ||| {}\n".format(num, translation,
                                                       str(cost))
                num += 1
                self.output_file.write(line)
                
    def print_json(self, beams, alignments, maxibatch, num_prev_translated):
        """
        Prints all available information as a JSON object.

        :param beams: n best translations each with their score, for a batch of inputs.
                      Shape: (batch, beam), individual beam items are (translation, score)
        :param alignments: soft attention scores between all source and target words.
                           Shape: (batch, num_models, beam, target_seq_len, source_seq_len)
        :param maxibatch: a list of input (source) sentences
        :param num_prev_translated: how many sentences were translated in this file before
                                    this specific maxibatch
        """
        outputs =[]

        for batch_index, (alignment, beam) in enumerate(zip(alignments, beams)):
            for beam_index, (translation, score) in enumerate(beam):

                translation = util.seq2words(translation, self.vocab)

                output = {
                    "idx":          num_prev_translated + batch_index,
                    "beam_index":   beam_index,
                    "translation":  translation + "<eos>",
                    "cost":         str(score)
                }
                outputs.append(output)

        json.dump(outputs, self.output_file, indent=4, ensure_ascii=False)
        
    def print_soft(self, beams, alignments, maxibatch, num_prev_translated):
        """
        Prints translation and soft attention weights in a line-based format.

        :param beams: n best translations each with their score, for a batch of inputs.
                      Shape: (batch, beam), individual beam items are (translation, score)
        :param alignments: soft attention scores between all source and target words.
                           Shape: (batch, num_models, beam, target_seq_len, source_seq_len)
        :param maxibatch: a list of input (source) sentences
        :param num_prev_translated: how many sentences were translated in this file before
                                    this specific maxibatch
        """
        for minibatch_index, (alignment, beam) in enumerate(zip(alignments, beams)):
            num = num_prev_translated + minibatch_index
            input_sentence = maxibatch[minibatch_index].rstrip()
            best_hypo, cost = beam[0]   ## TODO: with nbest?
            translation = util.seq2words(best_hypo, self.vocab)
            model_idx = 0 # TODO: more than one model
            beam_idx = 0 # TODO: nbest
            
            self.output_file.write("{} ||| {} ||| {} ||| {} ||| {} {}".format(num, translation, cost, input_sentence, len(input_sentence.split())+1, len(translation.split())+1))
            self.output_file.write("\n")
            
            # one line per target word
            for trg_idx in range(len(translation.split())+1):
                # print score for each source word
                for src_idx in range(len(input_sentence.split())+1):
                    self.output_file.write(str(alignment[model_idx, beam_idx, trg_idx, src_idx]) + " ")
                self.output_file.write("\n")
            self.output_file.write("\n")
