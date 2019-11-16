import numpy as np

def transform_letter_to_index(transcript, letter_list):
	'''
	:param transcript : Transcripts are the text input
	:param letter_list : Letter list defined above
	:return letter_to_index_list : Returns a list for all the transcript sentence to index
	'''
	letter_to_index_list = []
	for utterance in transcript:
		letters = []
		letters.append(letter_list.index('<sos>'))
		for byte in utterance:
			string = byte.decode('utf8')
			
			for character in string:
				letters.append(letter_list.index(character))

		letters.append(letter_list.index('<eos>'))
		letter_to_index_list.append(letters)

	return letter_to_index_list

