library(ggplot2)
ggplot(data.frame(x=c(0, 1)), aes(x=x)) +
  stat_function(fun=function(x){-(x*log2(x) + (1-x)*log2((1-x)/4))},
                size=2,
                aes(color = "line1")) +
  stat_function(fun=function(x){-log2(x^2 + 1/4*(1-x)^2)},
                size=2,
                aes(color = "line2")) +
  labs(x=expression(italic(x)),
       y="",
       title= "Entropy for Distribution\t\t\t")+
  theme_minimal() +
  theme(text=element_text(size=16,family="serif"),
        plot.title = element_text(hjust = 0.5, face="bold"),
        axis.title.x.bottom = element_text(size=16, family="serif")) +
  scale_color_manual(name="Entropy",
                     labels=c(line1="Shannon",
                              line2="Collision"),
                     values=c("red","blue"))