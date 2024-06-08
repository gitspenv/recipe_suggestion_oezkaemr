import streamlit as st
import pandas as pd
from ultralytics import YOLO
import os
import cv2
import json

# Paths
model_path = r'C:\Users\emreo\Desktop\ZHAW\ML2\project\models\best.pt'
test_images_path = r'C:\Users\emreo\Desktop\ZHAW\ML2\project\test'
recipes_data_path = r'C:\Users\emreo\Desktop\ZHAW\ML2\project\recipes_data.csv'

# Load the trained model
model = YOLO(model_path)

# Ingredient names corresponding to class indices
ingredient_names = [
    'Akabare Chili', 'Apple', 'Artichoke', 'Winter Melon', 'Asparagus', 
    'Avocado', 'Bacon', 'Bamboo Shoots', 'Banana', 'Beans', 'Beaten Rice', 
    'Beef', 'Beetroot', 'Amaranth Greens', 'Bitter Gourd', 'Black Lentils', 'Black Beans', 
    'Bottle Gourd', 'Bread', 'Eggplant', 'Broad Beans', 'Broccoli', 'Buffalo Meat', 
    'Butter', 'Cabbage', 'Bell Pepper', 'Carrot', 'Cassava', 'Cauliflower', 'Chayote', 
    'Cheese', 'Chicken Gizzards', 'Chicken', 'Chickpeas', 'Chili Pepper', 'Chili Powder', 
    'Chow Mein Noodles', 'Cinnamon', 'Coriander', 'Corn', 'Cornflakes', 'Crab Meat', 
    'Cucumber', 'Egg', 'Pumpkin Leaves', 'Fiddlehead Ferns', 'Fish', 'Garden Peas', 
    'Garden Cress', 'Garlic', 'Ginger', 'Green Eggplant', 'Green Lentils', 
    'Mint', 'Green Peas', 'Green Soybeans', 'Fermented Leafy Greens', 'Ham', 
    'Ice', 'Jackfruit', 'Ketchup', 'Nepali Hog Plum', 'Lemon', 'Lime', 
    'Long Beans', 'Sun-Dried Lentil Balls', 'Milk', 'Minced Meat', 'Drumstick Leaves', 
    'Mushroom', 'Mutton', 'Soy Chunks', 'Okra', 'Olive Oil', 'Spring Onions', 
    'Onion', 'Orange', 'Indian Spinach', 'Nepali Spinach', 'Paneer', 'Papaya', 
    'Pea', 'Pear', 'Pointed Gourd', 'Pork', 'Potato', 'Pumpkin', 'Radish', 
    'Pigeon Peas', 'Mustard Greens', 'Red Beans', 'Red Lentils', 'Rice', 'Drumsticks', 
    'Salt', 'Sausage', 'Snake Gourd', 'Soy Sauce', 'Soybeans', 'Sponge Gourd', 
    'Stinging Nettle', 'Strawberry', 'Sugar', 'Sweet Potato', 'Taro Leaves', 
    'Taro Root', 'Thukpa Noodles', 'Tofu', 'Tomato', 'Mustard Greens', 'Tomato', 
    'Turnip', 'Walnut', 'Watermelon', 'Wheat', 'Yellow Lentils', 'Kimchi', 'Mayonnaise', 'Noodles', 'Seaweed'
]

# Load recipes data
recipes_df = pd.read_csv(recipes_data_path)

# Proper formatting of Links
base_url = "http://www.cookbooks.com/Recipe-Details.aspx?id="
recipes_df['link'] = recipes_df['link'].apply(lambda x: x if x.startswith('http') else base_url + x.split('=')[-1])

# Function to count ingredient matches and ensure distinct matches
def count_ingredient_matches(ingredients_list, detected_ingredients):
    matches = 0
    matched_ingredients = set()
    for ingredient in ingredients_list:
        for detected in detected_ingredients:
            if detected.lower() in ingredient.lower() or ingredient.lower() in detected.lower():
                matched_ingredients.add(detected)
                break
    matches = len(matched_ingredients)
    return matches, list(matched_ingredients)

# Streamlit app
st.title("Recipe Suggestion App")
st.write("Upload an image to get recipe suggestions based on detected ingredients.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    img_path = os.path.join(test_images_path, uploaded_file.name)
    
    # Save uploaded image
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Make predictions
    results = model(img_path)
    
    # Initialize an empty list to hold class names for this image
    all_class_names = set()  # Use a set to ensure distinct values
    
    # Convert class indices to ingredient names
    predictions = []
    for result in results:
        for pred in result.boxes.data.tolist():
            box = pred[:4]
            score = pred[4]
            class_idx = int(pred[5])
            class_name = ingredient_names[class_idx]
            predictions.append({
                'box': box,
                'score': score,
                'class': class_name
            })
            all_class_names.add(class_name)
    
    # Convert set to list for further processing
    all_class_names = list(all_class_names)
    
    # Read the image using OpenCV
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display the image
    st.image(image, caption='Processed Image', use_column_width=True)
    
    # Display predicted ingredients
    st.subheader("Predicted Ingredients")
    st.write(", ".join(all_class_names))
    
    # Function to check for ingredient matches
    def ingredient_match(ingredients_list, detected_ingredients):
        matches = 0
        for detected in detected_ingredients:
            if any(detected.lower() in ingredient.lower() for ingredient in ingredients_list):
                matches += 1
        return matches == len(detected_ingredients)

    # Suggest recipes based on detected ingredients
    suggested_recipes = []
    if len(all_class_names) > 2:
        for _, row in recipes_df.iterrows():
            recipe_ingredients = json.loads(row['ingredients'])
            if ingredient_match(recipe_ingredients, all_class_names):
                if len(recipe_ingredients) <= 15:  # Filter by maximum number of ingredients
                    suggested_recipes.append({
                        'title': row['title'],
                        'ingredients': recipe_ingredients,
                        'directions': json.loads(row['directions']),
                        'link': row['link']
                    })

    # Output suggested recipes
    st.subheader("Suggested Recipes")
    if suggested_recipes:
        # Display recipes with all detected ingredients
        st.write("Recipes with all detected ingredients:")
        for recipe in suggested_recipes[:10]:  # Limit to 10 recipes
            st.markdown(f"### {recipe['title']}")
            st.markdown("#### Ingredients")
            st.markdown("\n".join([f"- {ingredient}" for ingredient in recipe['ingredients']]))
            st.markdown("#### Instructions")
            st.markdown("\n".join([f"{i+1}. {step}" for i, step in enumerate(recipe['directions'])]))
            st.markdown(f"[Recipe Link]({recipe['link']})")
            st.markdown("---")
    else:
        # Check if there are at least two detected ingredients
        if len(all_class_names) >= 2:
            # Display recipes with at least two detected ingredients
            st.write("No recipes found with all detected ingredients. Showing recipes with at least two detected ingredients:")
            suggested_recipes = []
            for _, row in recipes_df.iterrows():
                recipe_ingredients = json.loads(row['ingredients'])
                match_count, matched_ingredients = count_ingredient_matches(recipe_ingredients, all_class_names)
                if match_count >= 2:
                    suggested_recipes.append({
                        'title': row['title'],
                        'ingredients': recipe_ingredients,
                        'directions': json.loads(row['directions']),
                        'link': row['link'],
                        'matched_ingredients': matched_ingredients
                    })

            if suggested_recipes:
                for recipe in suggested_recipes[:10]:  # Limit to 10 recipes
                    st.markdown(f"### {recipe['title']}")
                    st.markdown("#### Matched Ingredients")
                    st.markdown(", ".join(recipe['matched_ingredients']))
                    st.markdown("#### Ingredients")
                    st.markdown("\n".join([f"- {ingredient}" for ingredient in recipe['ingredients']]))
                    st.markdown("#### Instructions")
                    st.markdown("\n".join([f"{i+1}. {step}" for i, step in enumerate(recipe['directions'])]))
                    st.markdown(f"[Recipe Link]({recipe['link']})")
                    st.markdown("---")
            else:
                # Display recipes for each detected ingredient
                st.write("No recipes found with at least two detected ingredients. Showing recipes with each detected ingredient:")
                for ingredient in all_class_names:
                    ingredient_recipes = []
                    for _, row in recipes_df.iterrows():
                        recipe_ingredients = json.loads(row['ingredients'])
                        if ingredient_match(recipe_ingredients, [ingredient]):
                            ingredient_recipes.append({
                                'title': row['title'],
                                'ingredients': recipe_ingredients,
                                'directions': json.loads(row['directions']),
                                'link': row['link']
                            })
                    st.markdown(f"### Recipes with {ingredient}:")
                    for recipe in ingredient_recipes[:5]:  # Limit to 5 recipes per ingredient
                        st.markdown(f"### {recipe['title']}")
                        st.markdown("#### Ingredients")
                        st.markdown("\n".join([f"- {ingredient}" for ingredient in recipe['ingredients']]))
                        st.markdown("#### Instructions")
                        st.markdown("\n".join([f"{i+1}. {step}" for i, step in enumerate(recipe['directions'])]))
                        st.markdown(f"[Recipe Link]({recipe['link']})")
                        st.markdown("---")
        else:
            # Display recipes for each detected ingredient
            st.write("No recipes found with at least two detected ingredients. Showing recipes with each detected ingredient:")
            for ingredient in all_class_names:
                ingredient_recipes = []
                for _, row in recipes_df.iterrows():
                    recipe_ingredients = json.loads(row['ingredients'])
                    if ingredient_match(recipe_ingredients, [ingredient]):
                        ingredient_recipes.append({
                            'title': row['title'],
                            'ingredients': recipe_ingredients,
                            'directions': json.loads(row['directions']),
                            'link': row['link']
                        })
                st.markdown(f"### Recipes with {ingredient}:")
                for recipe in ingredient_recipes[:5]:  # Limit to 5 recipes per ingredient
                    st.markdown(f"### {recipe['title']}")
                    st.markdown("#### Ingredients")
                    st.markdown("\n".join([f"- {ingredient}" for ingredient in recipe['ingredients']]))
                    st.markdown("#### Instructions")
                    st.markdown("\n".join([f"{i+1}. {step}" for i, step in enumerate(recipe['directions'])]))
                    st.markdown(f"[Recipe Link]({recipe['link']})")
                    st.markdown("---")
