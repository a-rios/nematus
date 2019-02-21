#!/usr/bin/env python

"""Prints different output formats."""

import logging

import tensorflow as tf

import inference
import util
import json
from collections import OrderedDict


class OutputHandler(object):
    def __init__(self, beams, alignments, outfile, output_format, maxibatch, vocab, num_prev_translated):
        self.beams = beams
        self.alignments = alignments
        self.outfile = outfile
        self.output_format = output_format
        self.maxibatch = maxibatch
        self.vocab = vocab
        self.num_prev_translated = num_prev_translated
        
        
    def write(self):
        if self.output_format == "translation":
            self.print_translation()
        elif self.output_format == "nbest":
            self.print_nbest()
        elif self.output_format == "json":
            self.print_json()
        elif self.output_format == "soft":
            self.print_soft()
            
    def get_words(self, line):
        words = line.split()
        words.append('<eos>')
        return words
        
    def print_translation(self):
        # Write the translations to the output file.
        for beam in self.beams:
            best_hypo, cost = beam[0]
            line = util.seq2words(best_hypo, self.vocab) + '\n'
            self.outfile.write(line)


    def print_nbest(self):
        for i, beam in enumerate(self.beams):
            num = self.num_prev_translated + i
            for sent, cost in beam:
                translation = util.seq2words(sent, self.vocab)
                line = "{} ||| {} ||| {}\n".format(num, translation,
                                                       str(cost))
                self.outfile.write(line)
                
    def print_json(self):
        sentences =[]
        for i, (alignment, beam) in enumerate(zip(self.alignments, self.beams)): # alignments shape (batch, num_models, beam, target_seq_len, source_seq_len)
                 num_models = alignment.shape[0] # alignment shape ( num_models, beam, target_seq_len, source_seq_len)
                 translation_maxlen = alignment.shape[2]
                 input_len = alignment.shape[3]
                 input_sentence = self.maxibatch[i].rstrip()
                 source_words = self.get_words(input_sentence)
                 
                 best_hypo, cost = beam[0]   ## TODO: with nbest?
                 line = util.seq2words(best_hypo, self.vocab)
                 sentence = OrderedDict ([
                        ('translation' , line),
                        ('source', input_sentence),
                        ('cost' , cost)
                        ])
                 words = self.get_words(line)
                 model_idx=0 # TODO: more than one model
                 best_beam_idx=0
                 score_sum=0
                 alignment_list = OrderedDict()
                 for target_word_idx, target_word in enumerate(words):
                    scores = OrderedDict()
                    for j, source_word in enumerate(source_words):
                        scores[source_word] = alignment[model_idx, best_beam_idx, target_word_idx, j].item()
                        alignment_list[target_word]=scores
                        sentence['alignments'] = alignment_list
                    sentences.append(sentence)
        json.dump(sentences, self.outfile, indent=4, ensure_ascii=False)
        
    def print_soft(self):
        for i, (alignment, beam) in enumerate(zip(self.alignments, self.beams)):
            input_sentence = self.maxibatch[i].rstrip()
            source_words = self.get_words(input_sentence)
            best_hypo, cost = beam[0]   ## TODO: with nbest?
            translation = util.seq2words(best_hypo, self.vocab)
            model_idx = 0 # TODO: more than one model
            beam_idx =0 # TODO: nbest
            
            self.outfile.write("{} ||| {} ||| {} ||| {} ||| {} {}".format(i, translation, cost, input_sentence, len(input_sentence.split())+1, len(translation.split())+1))
            self.outfile.write("\n")
            
            # one line per target word
            for trg_idx in range(len(translation.split())+1):
                # print score foreach source word
                scores = ""
                for src_idx in range(len(input_sentence.split())+1):
                    self.outfile.write(str(self.alignments[i, model_idx, beam_idx, trg_idx, src_idx]) + " ")
                self.outfile.write("\n")    
            self.outfile.write("\n")
                
