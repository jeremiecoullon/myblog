---
layout: post
title: "The DjangoVerse"
date: 2019-10-29 18:01:52 +0000
categories: programming design gypsyjazz
---

The [DjangoVerse](https://www.londondjangocollective.com/djangoverse/) is a 3D graph of gypsy jazz players around the world. I designed this with [Matt Holborn](https://www.mattholborn.com) (he got the idea from [the Rhizome](https://www.coreymwamba.co.uk/resources/rhizome/)) and built it using React and Django.

## How does it work ?

As anyone can modify it, people can [add themselves or players](https://www.londondjangocollective.com/djangoverse/forms/player/list) they know to it. If you click on a player you get information about them: what instrument they play, a picture of them, a short bio, and a link to a youtube video of them. As the names are coloured by country, you can immediately see how many players there are in the different countries around the world. You can try out the DjangoVerse in the figure below:


<figure style="text-align:center">
  <iframe src="https://djangoversereact.s3.eu-west-2.amazonaws.com/index.html" style="width:96%; margin-left:2%; height:400px;"></iframe>
  <figcaption><a href="https://www.londondjangocollective.com/djangoverse/">The DjangoVerse</a></figcaption>
</figure>

The players have a link between them if they have gigged together, and if you click on a player you get those links highlighted in red. This allows you to see at a glance who they've played with and whether they've played with people from different countries. You can also filter the graph to only display players from chosen countries, based on the instruments they play, or whether or not they're active. We started out by added around 60 players ourselves, and then shared it on Facebook and Instagram; the gypsy jazz community added the rest (there are 220 players across 21 countries at the time of writing).

## Tech stack

I built the graph with React and [D3 force directed graph](https://github.com/vasturiano/react-force-graph) and hosted it on S3 ([see code](https://github.com/jeremiecoullon/DjangoVerse-react)). The API is built using Django and Postgres and is hosted on Heroku (with S3 for static files). As the DjangoVerse is part of the [London Django Collective](https://www.londondjangocollective.com/), I used the same [Django application](https://github.com/jeremiecoullon/ldc) to serve the pages for the Collective as well as the API. As the React app with the graph is hosted on S3, the [page](https://www.londondjangocollective.com/djangoverse/) in the Collective website simply has an iframe that points to it.


# The design process

## A first attempt

The main motivation was that I've wanted for a long time to create a 3D graph mapping links between related things (and had ideas about doing this for academic disciplines, jazz standards, and more). So this project was a way to scratch that itch. The objective more specifically was to be able to visualise the gypsy jazz scene in one place, discover new players and bands, and let people be able to promote their music/bands.

As a result we started off with many different types of nodes: players, bands, festivals, albums, and venues. So each of these would be added to the graph along with links between them. A link between a player and band would mean that a players is in a band, a link between a band and a festival would mean that it's played at the festival, and so on. Each node would be a sphere of different size (the size would depend on the type) and the name would appear on hover; this was inspired by [Steemverse](https://steemverse.com/) (a visualisation of a social network).

Furthermore, the links between two nodes would also have information about it, such as the year a band has played in a festival, or the years a player was active in a band. You would then be able to filter the graph to only show what happened in a given year, which would give a "snapshot" of the gypsy jazz scene at that moment in time.

## Too much stuff

However, it quickly became clear that it was too much information: having all these types of nodes and information about the links would be too overwhelming to have in the graph. So we removed the venue and album types, along with the information about each link. We kept only the active/inactive tags which would allow to differentiate between the gypsy jazz scene in past and in the present.

We then tested a prototype (with players, bands, and venues all represented as spheres of different sizes) with some friends (see the classic [Don't Make Me Think](https://www.amazon.co.uk/Dont-Make-Me-Think-Usability/dp/0321344758) for an overview of user testing), and it turned out that it wasn't very clear what the DjangoVerse was. For example one reaction was _"I'm guessing it's a simulation of a molecule or something"_, which makes sense given that it essentially looked like [this](https://vasturiano.github.io/3d-force-graph/example/async-load/). This could maybe be fixed by adding names next to the nodes, but if you do this then D3 starts lagging quite quickly as you add many players.

Another problem was that festivals naturally ended up being at the centre of the graph, as they were the nodes with the most connections. The players and bands themselves then ended up seeming less important, even though we think a style of music is mainly about the players themselves rather than the festivals. As a visualisation is supposed to bring out the aspects of the data that the designer thinks is most important, we needed to have the players be more prominent.

## Simplifying the design

A fix to both of these problems was to simplify the graph again: we remove festivals and albums and kept just the players. We also just showed the names of the players rather than the spheres. As the names are immediately visible, a user can then recognise some of the players and guess immediately what this is about (this was confirmed with testing). However a downside of this is that having all the names rather than just spheres causes the graph to lag when there are more than 100 or so players. [Steemverse](https://steemverse.com/) gets around this problem by only having names for the "category" types of nodes (which are rare); all other spheres only have names on hover.

For the aspect of users adding players, there is no authentication so anyone can add or modify a player without needing to log in. The benefit is that there is less of a barrier for people to add to the graph, but with the risk of people posting spam (or deleting all the players!). To mitigate this, I set up daily backups (easy to do with Heroku) which would allow to restore the graph to before there was a problem. If the problem persisted, I would have simply added authentication (for example OAuth with Google/Facebook/etc..).

# Outcomes and comparison to other graphs

Players on the gypsy jazz scene around the world added lots of players to the graph: there are 220 players spanning 21 countries and with 9 instruments represented. A feature that was used a lot was the possibility of adding a youtube video: this allows each player to showcase their music. The short bio for each player was also interesting; when we added the bio we didn't think much of it nor consider too much how it would be used. However some of the users added information such as which players were related to each other (father, cousin etc..) which was really interesting!

## Lessons

In terms of design, an important take-away to be learnt from graph visualisations such as this is about how much information to include in it. Although a main aspect of these visualisations is just "eye-candy" (ie: it looks fun), it would be good if it was also informative or insightful. At one end of the spectrum, if there is too little information then there is not much to learn from the visualisation. At the other extreme, if there is too much information (and the design isn't done carefully) then it's easy to get overwhelmed. For me, some examples of this are [Wikiverse](https://www.wikiverse.io/) (it has a huge amount of information (it's a subset of wikipedia!) and I find the interface very confusing), [Steemverse](https://steemverse.com/) (it looks great, but there's not much information in it) or the [Rhizome](https://www.coreymwamba.co.uk/resources/rhizome/) (as it's in only 2 dimensions, it's hard to see what's going on in the graph).


In contrast, an example of a simple graph that I think works well is this [map of "theories of everything"](https://www.quantamagazine.org/frontier-of-physics-interactive-map-20150803/). I don't understand what these theories are (these are disciplines in theoretical physics), but the design is done very well and classifies them in a clear way.


Other examples of very well designed graphs are the ones built by [concept.space](http://concept.space/), such as this [map of philosophy](http://map.philosophies.space/). It has a huge amount of information, but most of it is hidden if you are zoomed out. As you zoom into a specific area of philosophy you get more and more detail about that area of philosophy until you have individual papers. When you click on a paper you then get the abstract and a link to it.


<!-- <figure style="text-align:center">
  <iframe src="http://map.philosophies.space/" style="width:96%; margin-left:2%; height:400px;"></iframe>
  <figcaption>Philosophies Space</figcaption>
</figure> -->

Notice also the minimap in the lower right hand corner that reminds you of where you currently are in the map. Finally, it seems that they have automated the process of adding and clustering the papers (from looking at the software [credited](http://philosophies.space/credits/) on their website). They seemed to have scraped [PhilPapers](https://philpapers.org/), used [Word2Vec](https://code.google.com/archive/p/word2vec/) to get word embeddings for each paper, [reduced the dimension](https://github.com/lmcinnes/umap) of the space, and finally [clustered](https://hdbscan.readthedocs.io/en/latest/) the result to find the location of each paper in the 2 dimensional map. As a result they could then use this workflow to create a similar map for [climate science](http://map.climate.space/) and [biomedicine](http://concept.space/projects/biomap/).

In conclusion, the idea of a visual map showing the links between different things in a discipline (players in gypsy jazz, papers in philosophy, etc..) is a very appealing one. However, getting it right is surprisingly difficult; for me the best example is the map of philosophy described above.
