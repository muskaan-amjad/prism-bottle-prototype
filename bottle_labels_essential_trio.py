import pandas as pd
import os

# Essential Trio functional feature scoring
bottle_categories = {
    'Wine Bottle': {
        'durability': 9,        # Glass strength
        'chemical_safety': 9,   # Food-grade materials
        'ergonomics': 6         # Elegant but less practical
    },
    'Water Bottle': {
        'durability': 7,        # Daily use strength
        'chemical_safety': 8,   # BPA-free requirements
        'ergonomics': 9         # Optimized for drinking
    },
    'Soda Bottle': {
        'durability': 6,        # Carbonation pressure
        'chemical_safety': 7,   # Acidic drink compatibility
        'ergonomics': 8         # Easy grip design
    },
    'Plastic Bottles': {
        'durability': 5,        # Variable quality
        'chemical_safety': 6,   # Plastic grade dependent
        'ergonomics': 8         # Lightweight design
    },
    'Beer Bottles': {
        'durability': 8,        # Thick glass
        'chemical_safety': 9,   # Alcohol compatibility
        'ergonomics': 7         # Traditional shape
    }
}

def create_bottle_labels():
    labels_data = []
    
    for category, features in bottle_categories.items():
        # Create labels for all 5000 images in each category
        for i in range(5000):
            image_filename = f"{i:08d}.jpg"  # Format: 00000000.jpg
            
            labels_data.append({
                'image_filename': image_filename,
                'category': category,
                'bottle_type': category.replace(' ', '_').lower(),
                'durability_score': features['durability'],
                'chemical_safety_score': features['chemical_safety'],
                'ergonomics_score': features['ergonomics']
            })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(labels_data)
    df.to_csv('bottle_labels_essential_trio.csv', index=False)
    
    print(f"✅ Created labels for {len(labels_data)} images")
    print(f"📁 Saved to: bottle_labels_essential_trio.csv")
    
    # Display summary
    print("\n📊 Dataset Summary:")
    print(df.groupby('category').size())
    
    print("\n🎯 Functional Feature Averages by Category:")
    feature_cols = ['durability_score', 'chemical_safety_score', 'ergonomics_score']
    print(df.groupby('category')[feature_cols].mean().round(1))
    
    return df

# Run the function
if __name__ == "__main__":
    df = create_bottle_labels()
    print("\n🎯 Sample entries:")
    print(df.head())
