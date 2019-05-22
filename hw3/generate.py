import torch
import rnn
from constants import *
from improvise_rnn.preprocess import *
from improvise_rnn.qa import *


def load_model(checkpoint_path):
  checkpoint = torch.load(checkpoint_path)
  model_name = checkpoint['model_name']
  hparams = checkpoint['hparams']

  model_class = getattr(rnn, model_name)
  model = model_class(**hparams)

  model.load_state_dict(checkpoint['model_state_dict'])

  return model


def generate_sequence(model, sequence_length):
  device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
  with torch.no_grad():
    model.eval()
    model = model.to(device)

    c_0, h_0 = model.init_hidden(batch_size=1, random_init=False)
    c_0 = c_0.to(device)
    h_0 = h_0.to(device)
    init_hidden = (c_0, h_0)
    init_output = torch.zeros((1,)).type(torch.LongTensor).to(device)

    hidden = init_hidden
    output = init_output
    # TODO: Fill in below
    # make empty list where output will be gathered
    outputs = None

    for step in range(sequence_length - 1):
      # TODO: Fill in below
      pred, hidden = None

      #######################################
      # we changed here to get better output results. (argmax sampling -> random sampling)
      # if you are interested, change this into argmax, and see the difference.
      # output = None  <- before

      out_dist = pred.data.view(-1).exp()
      output = torch.multinomial(out_dist, 1)
      #######################################

      outputs.append(output.cpu().numpy()[0])

    # TODO: Fill in below
    # return generated sequence
    return None


def indices_to_midi(indices, save_path=None, init_tempo=130):
  """
  Generates music using a model trained to learn musical patterns of a jazz soloist. Creates an audio stream

  Returns:
  predicted_tones -- python list containing predicted tones
  """

  chords, abstract_grammars = get_musical_data('data/original_metheny.mid')
  corpus, tones, tones_indices, indices_tones = get_corpus_data(abstract_grammars)

  # set up audio stream
  out_stream = stream.Stream()

  # Initialize chord variables
  curr_offset = 0.0  # variable used to write sounds to the Stream.
  num_chords = int(len(chords) / 3)  # number of different set of chords

  print("Predicting new values for different set of chords.")
  # Loop over all 18 set of chords. At each iteration generate a sequence of tones
  # and use the current chords to convert it into actual sounds
  for i in range(1, num_chords):
    # Retrieve current chord from stream
    curr_chords = stream.Voice()

    # Loop over the chords of the current set of chords
    for j in chords[i]:
      # Add chord to the current chords with the adequate offset, no need to understand this
      curr_chords.insert((j.offset % 4), j)
    pred = [indices_tones[p] for p in indices]

    predicted_tones = 'C,0.25 '
    for k in range(len(pred) - 1):
      predicted_tones += pred[k] + ' '

    predicted_tones += pred[-1]

    #### POST PROCESSING OF THE PREDICTED TONES ####
    # We will consider "A" and "X" as "C" tones. It is a common choice.
    predicted_tones = predicted_tones.replace(' A', ' C').replace(' X', ' C')

    # Pruning #1: smoothing measure
    predicted_tones = prune_grammar(predicted_tones)

    # Use predicted tones and current chords to generate sounds
    sounds = unparse_grammar(predicted_tones, curr_chords)

    # Pruning #2: removing repeated and too close together sounds
    sounds = prune_notes(sounds)

    # Quality assurance: clean up sounds
    sounds = clean_up_notes(sounds)

    # Print number of tones/notes in sounds
    print('Generated %s sounds using the predicted values for the set of chords ("%s") and after pruning' % (
      len([k for k in sounds if isinstance(k, note.Note)]), i))

    # Insert sounds into the output stream
    for m in sounds:
      out_stream.insert(curr_offset + m.offset, m)
    for mc in curr_chords:
      out_stream.insert(curr_offset + mc.offset, mc)

    curr_offset += 4.0

  # Initialize tempo of the output stream with 130 bit per minute
  out_stream.insert(0.0, tempo.MetronomeMark(number=init_tempo))

  # Save audio stream to fine
  mf = midi.translate.streamToMidiFile(out_stream)
  if save_path == None:
    save_path = 'output/my_music.midi'
  mf.open(save_path, 'wb')
  mf.write()
  print("Your generated music is saved in {}".format(save_path))
  mf.close()

  # Play the final stream through output (see 'play' lambda function above)
  # play = lambda x: midi.realtime.StreamPlayer(x).play()
  # play(out_stream)

  return out_stream


if __name__ == '__main__':
  model = load_model("runs/improvise_RNN_190515-100339/model-128.pt")
  seqs = generate_sequence(model, 30)
  print(seqs)
  indices_to_midi(seqs, save_path='output/my_music.mid', init_tempo=130)

  # to generate random sequence
  # seq_rand = list(np.random.randint(low=0, high=N_DICT, size=(50)))
  # indices_to_midi(seq_rand, 'output/random.mid')
