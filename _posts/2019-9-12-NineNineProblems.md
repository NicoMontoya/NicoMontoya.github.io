---
layout: post
title: 99 Problems
subtitle: Taking the Thinking out of Your Next Unicorn Idea
image: 
published: True
---
Project Link: https://ninenineproblems.com

**The Purpose**

The purpose of this project is to help Lambda School generate ideas for their students to work on. As Lambda School continues to grow, it has become difficult to keep up with idea generation for projects. That is where we come in. Our team has made use of leading edge technology in text generation to help create company descriptions. Using a data set of 150,000 company descriptions and training on top of OpenAI’s GPT-2 neural network, we were able to generate company descriptions that have the potential to be used by people in the real world.

**The Data** 

Before we can train on any model we needed to gather data, and in data science the old adage holds true - “Garbage in, garbage out”. That means we needed clean descriptions of problems and solutions so we could generate new ideas. You know what fits that mold? Company descriptions. So we gathered our data, 150,000 company descriptions, from Crunchbase. After some minor cleaning we can feed that into GPT-2 to generate novel company descriptions.

**The Model** 

GPT-2 is OpenAI’s text generation model that is trained using 40GB of text from the internet. They provided a smaller model than the original GPT-2 so that researchers can work and build on top of it, which is what our team used to transfer and further train with our company description dataset. The easy version is we show it a lot of examples of what we are looking for and it tries to imitate those examples as output.

**Post Processing**

When we use GPT-2 to generate novel company descriptions, the signal to noise ratio still isn’t where we want it to be. We end up having to sift through a lot of garbage to find a gem. To save us some effort and avoid manually trash diving we apply two post processing filters.

To understand how they work, let’s take a look at this terrible description:

*CITY of Design is a UK company with offices in London, London, London, and Paris. The company was established in 1864 and is headquartered in London, England.*

The first filter deals mainly with word frequency. We found that there is a goldilocks zone of word repetition where the descriptions start to make more sense. However in this zone, we still see descriptions like above, so we penalize a list of stop words that encode information that doesn’t matter to us. For instance, we don’t care where a fake company is based or when it was founded.

Once we do a pass for word frequency, we apply a second filter looking to identify entities. What is an entity? A real world object or concept such as time, location, or a person. We penalize descriptions with entity types we don’t care about - all we want is a problem and solution not a corporate history. Any descriptions that have a score indicating they may bore us with the useless get dumped, and then it’s all brought here.

**The Future**

You’ll notice that some of the random ideas we serve up after all this filtering are still nonsense while other company descriptions sound rather legitimate. We still have some crud, which is where this site comes in. By upvoting and downvoting company descriptions you help us label and filter descriptions so that future iterations of the project will be even better.
Don’t want to wait for our next product cycle? Head over to the leaderboard to see what’s already made it to the top.



