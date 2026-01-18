"""
E-commerce Personal Shopper - Mock Data

Mock product database with various retailers and categories.
In production, this would connect to real APIs.
"""

from typing import Optional
import random
from datetime import datetime, timedelta

from ..models import (
    Product, Deal, Review, ReviewSummary, Category, Retailer, Size,
    ProductVariant, PriceComparison
)


class MockProductDatabase:
    """
    Mock product database for demo purposes.
    
    In production, replace with real retail APIs:
    - Amazon Product Advertising API
    - Walmart Open API
    - Target API
    - etc.
    """
    
    def __init__(self):
        self.products = self._create_products()
        self.deals = self._create_deals()
        self.reviews = self._create_reviews()
    
    def _create_products(self) -> list[Product]:
        """Create mock product catalog."""
        products = []
        
        # Winter Jackets
        products.extend([
            Product(
                product_id="PAT-001",
                name="Nano Puff Jacket",
                brand="Patagonia",
                category=Category.OUTDOOR,
                subcategory="winter_jackets",
                price=199.00,
                original_price=229.00,
                retailer=Retailer.REI,
                description="Lightweight, windproof, warm insulated jacket with recycled materials",
                features=["Windproof", "Water-resistant", "Packable", "Recycled insulation"],
                materials=["100% recycled polyester", "60g PrimaLoft Gold insulation"],
                rating=4.8,
                review_count=2340,
                is_sustainable=True,
                sustainability_certifications=["Recycled Materials", "Fair Trade", "bluesign"],
                available_sizes=["XS", "S", "M", "L", "XL"],
                available_colors=["Black", "Navy", "Green", "Red"],
                in_stock=True,
                ships_free=True,
                tags=["winter", "sustainable", "packable", "layering"],
            ),
            Product(
                product_id="REI-001",
                name="Stormhenge 850 Down Jacket",
                brand="REI Co-op",
                category=Category.OUTDOOR,
                subcategory="winter_jackets",
                price=179.00,
                retailer=Retailer.REI,
                description="Premium waterproof down jacket with 850-fill power",
                features=["Waterproof", "850-fill down", "Helmet-compatible hood", "Pit zips"],
                materials=["Recycled nylon shell", "850-fill RDS down"],
                rating=4.6,
                review_count=892,
                is_sustainable=True,
                sustainability_certifications=["bluesign certified", "Responsible Down Standard"],
                available_sizes=["XS", "S", "M", "L", "XL", "XXL"],
                available_colors=["Black", "Blue", "Olive"],
                in_stock=True,
                ships_free=True,
                tags=["winter", "waterproof", "sustainable", "down"],
            ),
            Product(
                product_id="COT-001",
                name="Fuego Down Hooded Jacket",
                brand="Cotopaxi",
                category=Category.OUTDOOR,
                subcategory="winter_jackets",
                price=185.00,
                retailer=Retailer.REI,
                description="Responsibly sourced down jacket with unique colorways",
                features=["800-fill down", "Unique colorways", "Packable", "DWR finish"],
                materials=["Recycled polyester", "800-fill RDS down"],
                rating=4.7,
                review_count=567,
                is_sustainable=True,
                sustainability_certifications=["Responsible Down Standard", "B Corp"],
                available_sizes=["XS", "S", "M", "L", "XL"],
                available_colors=["Del Dia (Multi)", "Black", "Saltwater"],
                in_stock=True,
                ships_free=True,
                tags=["winter", "colorful", "sustainable", "down"],
            ),
            Product(
                product_id="NF-001",
                name="ThermoBall Eco Jacket",
                brand="The North Face",
                category=Category.OUTDOOR,
                subcategory="winter_jackets",
                price=220.00,
                original_price=250.00,
                retailer=Retailer.NORDSTROM,
                description="Synthetic insulation that maintains warmth when wet",
                features=["ThermoBall insulation", "Packable", "Water-resistant"],
                materials=["100% recycled fabrics", "ThermoBall Eco insulation"],
                rating=4.5,
                review_count=1205,
                is_sustainable=True,
                sustainability_certifications=["Recycled Materials"],
                available_sizes=["S", "M", "L", "XL"],
                available_colors=["Black", "Navy", "Red"],
                in_stock=True,
                ships_free=True,
                tags=["winter", "synthetic", "packable", "sustainable"],
            ),
            Product(
                product_id="COL-001",
                name="Powder Lite Hooded Jacket",
                brand="Columbia",
                category=Category.OUTDOOR,
                subcategory="winter_jackets",
                price=110.00,
                original_price=150.00,
                retailer=Retailer.AMAZON,
                description="Affordable water-resistant insulated jacket",
                features=["Omni-Heat reflective", "Water-resistant", "Synthetic insulation"],
                materials=["Polyester shell", "Synthetic insulation"],
                rating=4.4,
                review_count=3450,
                is_sustainable=False,
                available_sizes=["S", "M", "L", "XL", "XXL"],
                available_colors=["Black", "Blue", "Green", "Red"],
                in_stock=True,
                ships_free=True,
                tags=["winter", "budget", "water-resistant"],
            ),
        ])
        
        # Running Shoes
        products.extend([
            Product(
                product_id="NIKE-001",
                name="Air Zoom Pegasus 40",
                brand="Nike",
                category=Category.FOOTWEAR,
                subcategory="running_shoes",
                price=130.00,
                retailer=Retailer.NIKE,
                description="Responsive cushioning for everyday runs",
                features=["Zoom Air units", "Breathable mesh", "Responsive cushioning"],
                materials=["Engineered mesh upper", "React foam midsole"],
                rating=4.7,
                review_count=5678,
                is_sustainable=False,
                available_sizes=["7", "8", "9", "10", "11", "12", "13"],
                available_colors=["Black/White", "Blue", "Grey"],
                in_stock=True,
                ships_free=True,
                tags=["running", "everyday", "cushioned"],
            ),
            Product(
                product_id="ALLB-001",
                name="Tree Dasher 2",
                brand="Allbirds",
                category=Category.FOOTWEAR,
                subcategory="running_shoes",
                price=135.00,
                retailer=Retailer.AMAZON,
                description="Sustainable performance running shoe",
                features=["SwiftFoam midsole", "Eucalyptus fiber upper", "Carbon neutral"],
                materials=["FSC-certified eucalyptus", "Sugarcane-based foam"],
                rating=4.3,
                review_count=890,
                is_sustainable=True,
                sustainability_certifications=["Carbon Neutral", "B Corp"],
                available_sizes=["7", "8", "9", "10", "11", "12"],
                available_colors=["Black", "White", "Green"],
                in_stock=True,
                ships_free=True,
                tags=["running", "sustainable", "carbon-neutral"],
            ),
        ])
        
        # Electronics
        products.extend([
            Product(
                product_id="SONY-001",
                name="WH-1000XM5 Wireless Headphones",
                brand="Sony",
                category=Category.ELECTRONICS,
                subcategory="headphones",
                price=348.00,
                original_price=399.99,
                retailer=Retailer.AMAZON,
                description="Industry-leading noise cancellation",
                features=["30-hour battery", "Industry-leading ANC", "Multipoint connection"],
                rating=4.6,
                review_count=12450,
                is_sustainable=False,
                available_colors=["Black", "Silver", "Midnight Blue"],
                in_stock=True,
                ships_free=True,
                tags=["headphones", "wireless", "noise-cancelling", "premium"],
            ),
            Product(
                product_id="APPLE-001",
                name="AirPods Pro (2nd Gen)",
                brand="Apple",
                category=Category.ELECTRONICS,
                subcategory="earbuds",
                price=249.00,
                retailer=Retailer.BEST_BUY,
                description="Active noise cancellation with spatial audio",
                features=["Active noise cancellation", "Spatial audio", "MagSafe charging"],
                rating=4.7,
                review_count=28900,
                in_stock=True,
                ships_free=True,
                tags=["earbuds", "wireless", "noise-cancelling", "apple"],
            ),
        ])
        
        # Home goods
        products.extend([
            Product(
                product_id="YETI-001",
                name="Rambler 20 oz Tumbler",
                brand="YETI",
                category=Category.HOME,
                subcategory="drinkware",
                price=38.00,
                retailer=Retailer.TARGET,
                description="Double-wall vacuum insulated tumbler",
                features=["Double-wall insulation", "Dishwasher safe", "No-sweat design"],
                materials=["18/8 stainless steel"],
                rating=4.9,
                review_count=45000,
                available_colors=["Black", "Navy", "White", "Seafoam", "Charcoal"],
                in_stock=True,
                ships_free=False,
                tags=["tumbler", "insulated", "durable"],
            ),
        ])
        
        return products
    
    def _create_deals(self) -> list[Deal]:
        """Create mock deals and coupons."""
        return [
            Deal(
                deal_id="DEAL-001",
                deal_type="coupon",
                code="WINTER20",
                description="20% off winter jackets",
                discount_percent=20,
                applicable_categories=[Category.OUTDOOR],
                valid_until=(datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"),
                is_active=True,
            ),
            Deal(
                deal_id="DEAL-002",
                deal_type="sale",
                description="REI Member 10% off",
                discount_percent=10,
                retailer=Retailer.REI,
                is_active=True,
            ),
            Deal(
                deal_id="DEAL-003",
                deal_type="coupon",
                code="SAVE15",
                description="15% off your first order",
                discount_percent=15,
                retailer=Retailer.NORDSTROM,
                is_active=True,
            ),
            Deal(
                deal_id="DEAL-004",
                deal_type="price_drop",
                description="Price drop alert - Sony headphones",
                discount_amount=50,
                applicable_brands=["Sony"],
                is_active=True,
            ),
            Deal(
                deal_id="DEAL-005",
                deal_type="coupon",
                code="NEWUSER10",
                description="10% off for new customers",
                discount_percent=10,
                minimum_purchase=50,
                is_active=True,
            ),
            Deal(
                deal_id="DEAL-006",
                deal_type="sale",
                description="Free shipping on orders $35+",
                retailer=Retailer.TARGET,
                minimum_purchase=35,
                is_active=True,
            ),
        ]
    
    def _create_reviews(self) -> dict[str, ReviewSummary]:
        """Create mock review summaries."""
        return {
            "PAT-001": ReviewSummary(
                product_id="PAT-001",
                average_rating=4.8,
                total_reviews=2340,
                rating_distribution={5: 1800, 4: 400, 3: 100, 2: 30, 1: 10},
                overall_sentiment="very positive",
                pros=["Incredibly warm for its weight", "Packs down small", "Great for layering"],
                cons=["Pricey", "Runs slightly small", "Limited color options"],
                most_helpful_positive="Perfect jacket for cold weather hiking. Warm, lightweight, and packable!",
                most_helpful_negative="Runs small - order a size up if you want to layer underneath.",
                fit_feedback="Runs slightly small",
            ),
            "REI-001": ReviewSummary(
                product_id="REI-001",
                average_rating=4.6,
                total_reviews=892,
                rating_distribution={5: 600, 4: 200, 3: 70, 2: 15, 1: 7},
                overall_sentiment="positive",
                pros=["Excellent waterproofing", "Very warm", "Great value for REI members"],
                cons=["A bit bulky", "Hood is large", "Takes time to dry"],
                most_helpful_positive="Best waterproof down jacket I've owned. Stayed dry in heavy rain.",
                fit_feedback="True to size",
            ),
            "COT-001": ReviewSummary(
                product_id="COT-001",
                average_rating=4.7,
                total_reviews=567,
                rating_distribution={5: 400, 4: 130, 3: 30, 2: 5, 1: 2},
                overall_sentiment="very positive",
                pros=["Unique colors stand out", "Very warm", "Ethical company"],
                cons=["Del Dia colors vary", "Not waterproof", "Zipper can snag"],
                most_helpful_positive="Love my unique colorway! Gets compliments everywhere.",
                fit_feedback="True to size",
            ),
        }
    
    def search_products(
        self,
        query: str = "",
        category: Optional[Category] = None,
        min_price: float = 0,
        max_price: float = float('inf'),
        brands: list[str] = None,
        sustainable_only: bool = False,
        in_stock_only: bool = True,
        retailers: list[Retailer] = None,
    ) -> list[Product]:
        """Search products with filters."""
        results = []
        
        query_lower = query.lower()
        
        for product in self.products:
            # Query match
            if query:
                match_fields = [
                    product.name.lower(),
                    product.brand.lower(),
                    product.description.lower(),
                    product.subcategory.lower(),
                    " ".join(product.tags).lower(),
                ]
                if not any(query_lower in field for field in match_fields):
                    continue
            
            # Category filter
            if category and product.category != category:
                continue
            
            # Price filter
            if product.price < min_price or product.price > max_price:
                continue
            
            # Brand filter
            if brands and product.brand.lower() not in [b.lower() for b in brands]:
                continue
            
            # Sustainability filter
            if sustainable_only and not product.is_sustainable:
                continue
            
            # Stock filter
            if in_stock_only and not product.in_stock:
                continue
            
            # Retailer filter
            if retailers and product.retailer not in retailers:
                continue
            
            results.append(product)
        
        return results
    
    def get_product(self, product_id: str) -> Optional[Product]:
        """Get product by ID."""
        for product in self.products:
            if product.product_id == product_id:
                return product
        return None
    
    def get_reviews(self, product_id: str) -> Optional[ReviewSummary]:
        """Get review summary for product."""
        return self.reviews.get(product_id)
    
    def get_deals_for_product(self, product: Product) -> list[Deal]:
        """Get applicable deals for a product."""
        applicable = []
        
        for deal in self.deals:
            if not deal.is_active:
                continue
            
            # Check retailer match
            if deal.retailer and deal.retailer != product.retailer:
                continue
            
            # Check brand match
            if deal.applicable_brands and product.brand not in deal.applicable_brands:
                continue
            
            # Check category match
            if deal.applicable_categories and product.category not in deal.applicable_categories:
                continue
            
            applicable.append(deal)
        
        return applicable
    
    def get_price_comparison(self, product_name: str, brand: str) -> PriceComparison:
        """Get price comparison across retailers."""
        # Find similar products across retailers
        similar = [p for p in self.products if brand.lower() in p.brand.lower() and product_name.lower() in p.name.lower()]
        
        prices = {}
        for product in similar:
            prices[product.retailer.value] = {
                "price": product.price,
                "url": product.url,
                "in_stock": product.in_stock,
            }
        
        # Find lowest
        if prices:
            lowest_retailer = min(prices.items(), key=lambda x: x[1]["price"])
            lowest_price = lowest_retailer[1]["price"]
            lowest_name = lowest_retailer[0]
        else:
            lowest_price = 0
            lowest_name = None
        
        return PriceComparison(
            product_name=f"{brand} {product_name}",
            prices=prices,
            lowest_price=lowest_price,
            lowest_price_retailer=Retailer(lowest_name) if lowest_name else None,
            price_trend="stable",
            average_price=sum(p["price"] for p in prices.values()) / len(prices) if prices else 0,
        )


# Global instance
product_db = MockProductDatabase()
