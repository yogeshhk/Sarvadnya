A simple proto chat bot using Neo4j. Neo4j is used as bot's memory.
requirements:
1- spacy.
2- rasa.
3- neo4j and its driver library for python.

This proto bot only can learn about restaurnats and users. It will save the facts into its memory and when you ask it can reterive.
For start fresh you can say "clear" (it clear the memory)
To end just say 'bye'

Example run:

(py36) [bash]:python chatbot.py
>  Ready!
clear
>  OK! I forget everything you said to me!
Hi
>  Hi
I am Amir and live in San Francisco
>  I also like to live in san francisco!
I will make sure I remember next time I see you!
I like Persian food
>  Great choice!
I will make sure I remember next time I see you!
I also like Chinese food
>  Great choice!
I will make sure I remember next time I see you!
I like Korean food as well
>  I am not sure what you mean
I meant I like Korean food
>  Great choice!
I will make sure I remember next time I see you!
I like Japanese food too
>  I also like japanese
I will make sure I remember next time I see you!
I am hungry
>  OH! Sorry I can't find any restaurant in san francisco. Please teach me about resturants there.
Anar is a Persian restaurant in San Francisco
>  Good to know!
I will make sure I remember next time I see you!
find me something
>  How about anar which is a persian restaurant in san francisco
Halo is a Japanese fast food
>  Good to know!
I will make sure I remember next time I see you!
Halo is in San Francisco
>  Good to know!
I will make sure I remember next time I see you!
FInd me something to eat
>  How about anar which is a persian restaurant in san francisco
Find me something else to eat
>  How about halo which is a japanese restaurant in san francisco
thanks
>  Sure!
Bye
Bye


TODO:
- It's inference is very limited.
- It can benefit from general knowledge graph.

