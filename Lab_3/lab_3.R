# Load required libraries
library(knitr)
library(dplyr)
library(RColorBrewer)
library(ggplot2)
library(cluster)
library(factoextra)

# Load customers data
customers <- read.csv("C:\\Users\\Arseny\\Desktop\\Programs\\Code\\R\\Lab_3\\Customers.csv", sep=";")
kable(head(customers), align = "c")

# Convert gender to a factor variable and remove customer_id column
customers$gender <- factor(customers$gender)
customers <- customers %>% 
  mutate(gender = as.numeric(gender)) %>% 
  select(-customer_id)
kable(head(customers), align = "c")

# Compute WSS for values of k ranging from 1 to 10
wss_values <- sapply(1:10, function(k) sum(kmeans(customers, k)$withinss))

# Create a data frame with the WSS values and cluster numbers
wss_df <- data.frame(k = 1:10, wss = wss_values)

# Plot the elbow curve with ggplot2
ggplot(wss_df, aes(x = k, y = wss)) +
  geom_line(color = "#0072B2", size = 1.5) +
  geom_point(color = "#0072B2", size = 3, shape = 19) +
  labs(x = "Number of clusters (K)", y = "Total within-cluster sum of squares", 
       title = "Elbow Curve for K-Means Clustering") +
  theme(plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
        axis.title.x = element_text(size = 14, face = "bold"),
        axis.title.y = element_text(size = 14, face = "bold"),
        axis.text = element_text(size = 12),
        axis.line = element_line(size = 1.2),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank())

# Calculate the slope of the line joining the first and last points of the curve
slope <- (wss_values[10] - wss_values[1]) / 9

# Calculate the distances of the remaining points to this line
distances <- sapply(2:9, function(i) {
  abs(wss_values[i] - wss_values[1] - i * slope) / sqrt(1 + slope ^ 2)
})

# Determine the optimal number of clusters based on the elbow point
k <- which.max(distances) + 1

# Print the optimal number of clusters
cat("Optimal number of clusters:", k)

# Cluster the data using k-means algorithm
set.seed(42)
pr_cluster <- kmeans(customers, center=k, nstart=20)

# Create a color palette for the clusters
palette <- brewer.pal(3, "Set2")

customers_tbl <- as_tibble(customers)

ggplot(customers_tbl, aes(x = age, y = spending_score, color = factor(pr_cluster$cluster), shape = factor(pr_cluster$cluster))) +
  geom_point(size = 4) +
  scale_color_brewer(palette = "Set1") +
  labs(x = "Age", y = "Spending Score", title = "Customer Segmentation by Age and Spending Score") +
  theme_bw()

#Calculation of distance matrix
distance_matrix <- dist(customers, method = "euclidean")

#Clustering
cluster_model <- hclust(distance_matrix, method = "average")

#Visualization of dendrogram and display of clusters
fviz_dend(cluster_model, k = k, cex = 0.5)

#Representation of clusters as a vector
cluster_vector <- cutree(cluster_model, k = k)

#Creation of scatter plot
df <- data.frame(age = customers$age, spending_score = customers$spending_score, cluster = factor(cluster_vector))
# Create graph
ggplot(df, aes(x = age, y = spending_score, color = cluster, shape = cluster)) +
  geom_point(size = 4) +
  scale_color_brewer(palette = "Set1") +
  labs(x = "Age", y = "Spending Score", title = "Customer Segmentation by Age and Spending Score") +
  theme_bw()


#Evaluate K-means clustering
kmeans_silhouette <- silhouette(pr_cluster$cluster, distance_matrix)
kmeans_wss <- sum(pr_cluster$withinss)

#Evaluate Hierarchical clustering
hierarchical_silhouette <- silhouette(cluster_vector, distance_matrix)
hierarchical_wss <- sum(cluster_model$height)

#Print evaluation metrics
cat("K-means clustering\n")
cat("Silhouette Width:", mean(kmeans_silhouette[,3]), "\n")
cat("Total within-cluster sum of squares:", kmeans_wss, "\n\n")

cat("Hierarchical clustering\n")
cat("Silhouette Width:", mean(hierarchical_silhouette[,3]), "\n")
cat("Total within-cluster sum of squares:", hierarchical_wss, "\n\n")

