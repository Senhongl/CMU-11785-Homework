import numpy as np

letter_list = ['<sos>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',\
             'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '.', '_', '+', ' ','<eos>']
			 
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

			letters.append(letter_list.index(' '))
		letters.append(letter_list.index('<eos>'))
		letter_to_index_list.append(letters)

	return letter_to_index_list

