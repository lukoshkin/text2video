# Video Generation Based on Short Text Description (2019)

Over a year latter, I have decided to add README to the repository, since
some people find the latter useful even without a description. I hope this
step will make the results of my work more usable for those who are interested
in the problem and stumble upon the repository when browsing the topic on GitHub.

## Example of Generated Video

Unfortunately, I have not saved videos generated by the network,
since all the results remained on the working laptop which I hand over
at the end of the internship. The only thing left is the recording
that I did on my cellphone (***Sorry if this makes your eyes bleed***).

What is on the gif? There are 5 blocks of images stacked horizontally.
Each block contains 4 objects, selected from '20bn-something-something-v2'
dataset and belonging to the same category _"Pushing [something] from left to right"_
<sup>**1**</sup> (~1000 samples). They are _book_ (top left window), _box_ (top right window),
_mug_ (bottom left window), and _marker_ (bottom right window), pushed along the surface by hand.
The number of occurrences in the data subset for the corresponding objects is 57, 43, 9, 55. 

The generated videos are diverse (thanks to [zero-gradient penalty](https://arxiv.org/abs/1902.03984))
and about the same quality as the videos from the training data. **There are no tests
conducted on the validation data.**

<p align="center">
  <img src="example.gif" />
</p>

---
<sup>**1**</sup> Yep, exactly "from left to right" and not the other way
around as you can read it on the gif (it is a typo). However, it is good
for validation purposes to make new labels with the reversed direction of
movement or new (but "similar", e.g., in space of embeddings) objects from unchanged category.

## Navigating Through SRC Files

<center>

<table>
<tbody>
  <tr>
    <td> data_prep2.py </th>
    <td> Video and text processing (based on text_processing.py) </td>
  </tr>
  <tr>
    <td> blocks.py </td>
    <td> Building blocks used in models.py </td>
  </tr>
  <tr>
    <td> visual_encoders.py </td>
    <td> Advanced building blocks for image and video discriminators </td>
  </tr>
  <tr>
    <td> process3-5.ipynb </td>
    <td> Pipeline for the training process on multiple gpus </br> (3-5 is a hardcoded range of gpus involved) </td>
  </tr>
  <tr>
    <td> pipeline.ipynb </td>
    <td> Previously served for the same purpose as pipeline.py </br> but hadrcoded range was 0-2.
         Now it is <b> unfinished implementation </b> of <a href='https://arxiv.org/abs/1806.07185'>mixed batches</a> </td>
  </tr>
  <tr>
    <td> legacy (=obsolete) </td>
    <td> Early attempts and ideas </td>
  </tr>
</tbody>
</table>

</center>

There is also a collection of [references](https://github.com/lukoshkin/text2video/blob/develop/references.md)
to articles relevant (at the time of 2019) to text2video generation problem.

## Dependencies

Dockerfile would be helpful here...  
Or I should have written this section earlier.

- torch 1.2.0
- nvidia/cuda 10.1
- at least one gpu available
- some other prerequisites?
