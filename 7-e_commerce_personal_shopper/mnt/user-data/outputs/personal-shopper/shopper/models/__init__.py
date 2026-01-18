"""
E-commerce Personal Shopper - Data Models

Models for:
- User preferences and profiles
- Products and variants
- Deals and coupons
- Reviews and ratings
- Recommendations
"""

from typing import Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json


class Category(Enum):
    CLOTHING = "clothing"
    ELECTRONICS = "electronics"
    HOME = "home"
    BEAUTY = "beauty"
    SPORTS = "sports"
    OUTDOOR = "outdoor"
    ACCESSORIES = "accessories"
    FOOTWEAR = "footwear"


class Retailer(Enum):
    AMAZON = "amazon"
    TARGET = "target"
    WALMART = "walmart"
    NORDSTROM = "nordstrom"
    REI = "rei"
    PATAGONIA = "patagonia"
    NIKE = "nike"
    BEST_BUY = "bestbuy"
    MACYS = "macys"
    ZAPPOS = "zappos"


class Size(Enum):
    XS = "XS"
    S = "S"
    M = "M"
    L = "L"
    XL = "XL"
    XXL = "XXL"


class PriceRange(Enum):
    BUDGET = "budget"         # Under $50
    MODERATE = "moderate"     # $50-150
    PREMIUM = "premium"       # $150-300
    LUXURY = "luxury"         # $300+


@dataclass
class UserPreferences:
    """User shopping preferences."""
    user_id: str = "default"
    
    # Style preferences
    preferred_styles: list[str] = field(default_factory=list)
    preferred_colors: list[str] = field(default_factory=list)
    preferred_brands: list[str] = field(default_factory=list)
    avoided_brands: list[str] = field(default_factory=list)
    
    # Sizing
    clothing_size: Optional[Size] = None
    shoe_size: Optional[str] = None
    
    # Values
    prefers_sustainable: bool = False
    prefers_local: bool = False
    prefers_ethical: bool = False
    
    # Budget
    default_budget: Optional[float] = None
    price_sensitivity: str = "moderate"  # low, moderate, high
    
    # Shopping behavior
    preferred_retailers: list[Retailer] = field(default_factory=list)
    avoided_retailers: list[Retailer] = field(default_factory=list)
    
    # History
    past_purchases: list[str] = field(default_factory=list)
    wishlisted_items: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "preferred_styles": self.preferred_styles,
            "preferred_colors": self.preferred_colors,
            "preferred_brands": self.preferred_brands,
            "prefers_sustainable": self.prefers_sustainable,
            "clothing_size": self.clothing_size.value if self.clothing_size else None,
            "default_budget": self.default_budget,
        }


@dataclass
class ProductVariant:
    """Product variant (size, color, etc.)."""
    variant_id: str
    size: Optional[str] = None
    color: Optional[str] = None
    price: float = 0.0
    in_stock: bool = True
    quantity_available: int = 0


@dataclass
class Product:
    """Product information."""
    product_id: str
    name: str
    brand: str
    
    # Categorization
    category: Category = Category.CLOTHING
    subcategory: str = ""
    tags: list[str] = field(default_factory=list)
    
    # Pricing
    price: float = 0.0
    original_price: Optional[float] = None
    currency: str = "USD"
    
    # Retailer
    retailer: Retailer = Retailer.AMAZON
    url: str = ""
    
    # Description
    description: str = ""
    features: list[str] = field(default_factory=list)
    materials: list[str] = field(default_factory=list)
    
    # Images
    image_url: str = ""
    additional_images: list[str] = field(default_factory=list)
    
    # Variants
    variants: list[ProductVariant] = field(default_factory=list)
    available_sizes: list[str] = field(default_factory=list)
    available_colors: list[str] = field(default_factory=list)
    
    # Ratings
    rating: float = 0.0
    review_count: int = 0
    
    # Sustainability
    is_sustainable: bool = False
    sustainability_certifications: list[str] = field(default_factory=list)
    
    # Availability
    in_stock: bool = True
    ships_free: bool = False
    estimated_delivery: str = ""
    
    @property
    def discount_percent(self) -> Optional[float]:
        if self.original_price and self.original_price > self.price:
            return ((self.original_price - self.price) / self.original_price) * 100
        return None
    
    @property
    def is_on_sale(self) -> bool:
        return self.original_price is not None and self.original_price > self.price
    
    def to_dict(self) -> dict:
        return {
            "product_id": self.product_id,
            "name": self.name,
            "brand": self.brand,
            "category": self.category.value,
            "price": self.price,
            "original_price": self.original_price,
            "retailer": self.retailer.value,
            "rating": self.rating,
            "review_count": self.review_count,
            "is_sustainable": self.is_sustainable,
            "in_stock": self.in_stock,
        }
    
    def to_display(self) -> str:
        """Format for display."""
        display = f"**{self.name}** by {self.brand}\n"
        
        if self.is_on_sale:
            display += f"💰 ${self.price:.2f} ~~${self.original_price:.2f}~~ ({self.discount_percent:.0f}% off)\n"
        else:
            display += f"💰 ${self.price:.2f}\n"
        
        display += f"⭐ {self.rating}/5 ({self.review_count:,} reviews)\n"
        display += f"🏪 {self.retailer.value.title()}\n"
        
        if self.is_sustainable:
            display += f"♻️ Sustainable: {', '.join(self.sustainability_certifications)}\n"
        
        if self.ships_free:
            display += "🚚 Free shipping\n"
        
        return display


@dataclass
class Deal:
    """Deal or coupon information."""
    deal_id: str
    
    # Type
    deal_type: str = "coupon"  # coupon, sale, price_drop, clearance
    
    # Details
    code: Optional[str] = None
    description: str = ""
    discount_amount: Optional[float] = None
    discount_percent: Optional[float] = None
    
    # Scope
    retailer: Optional[Retailer] = None
    applicable_brands: list[str] = field(default_factory=list)
    applicable_categories: list[Category] = field(default_factory=list)
    
    # Requirements
    minimum_purchase: Optional[float] = None
    
    # Validity
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    is_active: bool = True
    
    # Usage
    usage_limit: Optional[int] = None
    times_used: int = 0
    
    def calculate_discount(self, price: float) -> float:
        """Calculate discounted price."""
        if self.minimum_purchase and price < self.minimum_purchase:
            return price
        
        if self.discount_amount:
            return max(0, price - self.discount_amount)
        elif self.discount_percent:
            return price * (1 - self.discount_percent / 100)
        
        return price
    
    def to_display(self) -> str:
        display = ""
        if self.code:
            display += f"🏷️ Code: **{self.code}** - "
        
        if self.discount_percent:
            display += f"{self.discount_percent:.0f}% off"
        elif self.discount_amount:
            display += f"${self.discount_amount:.2f} off"
        
        display += f" | {self.description}"
        
        if self.valid_until:
            display += f" (expires {self.valid_until})"
        
        return display


@dataclass
class Review:
    """Product review."""
    review_id: str
    product_id: str
    
    # Content
    rating: int = 5  # 1-5
    title: str = ""
    text: str = ""
    
    # Reviewer
    reviewer_name: str = "Anonymous"
    verified_purchase: bool = False
    
    # Metadata
    date: str = ""
    helpful_votes: int = 0
    
    # Analysis
    sentiment: str = "positive"  # positive, negative, neutral
    key_points: list[str] = field(default_factory=list)


@dataclass
class ReviewSummary:
    """Summarized product reviews."""
    product_id: str
    
    # Aggregate
    average_rating: float = 0.0
    total_reviews: int = 0
    rating_distribution: dict = field(default_factory=dict)  # {5: 100, 4: 50, ...}
    
    # Analysis
    overall_sentiment: str = "positive"
    
    # Key themes
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)
    
    # Highlights
    most_helpful_positive: Optional[str] = None
    most_helpful_negative: Optional[str] = None
    
    # Fit info (for clothing)
    fit_feedback: str = ""  # "runs small", "true to size", "runs large"
    
    def to_display(self) -> str:
        display = f"⭐ {self.average_rating}/5 based on {self.total_reviews:,} reviews\n\n"
        
        display += "**What people love:**\n"
        for pro in self.pros[:3]:
            display += f"  ✅ {pro}\n"
        
        display += "\n**Things to consider:**\n"
        for con in self.cons[:3]:
            display += f"  ⚠️ {con}\n"
        
        if self.fit_feedback:
            display += f"\n👕 Fit: {self.fit_feedback}"
        
        return display


@dataclass
class PriceComparison:
    """Price comparison across retailers."""
    product_name: str
    
    # Prices by retailer
    prices: dict = field(default_factory=dict)  # {retailer: {price, url, in_stock}}
    
    # Best price
    lowest_price: float = 0.0
    lowest_price_retailer: Optional[Retailer] = None
    
    # Price history
    price_history: list[dict] = field(default_factory=list)  # [{date, price, retailer}]
    
    # Insights
    price_trend: str = "stable"  # rising, falling, stable
    average_price: float = 0.0
    
    def to_display(self) -> str:
        display = f"**Price Comparison: {self.product_name}**\n\n"
        
        sorted_prices = sorted(self.prices.items(), key=lambda x: x[1].get('price', 999999))
        
        for i, (retailer, info) in enumerate(sorted_prices):
            price = info.get('price', 0)
            in_stock = info.get('in_stock', True)
            
            if i == 0:
                display += f"🏆 **{retailer}**: ${price:.2f}"
            else:
                display += f"   {retailer}: ${price:.2f}"
            
            if not in_stock:
                display += " (Out of stock)"
            
            display += "\n"
        
        if self.price_trend == "falling":
            display += "\n📉 Price trend: Falling (good time to buy!)"
        elif self.price_trend == "rising":
            display += "\n📈 Price trend: Rising (consider buying soon)"
        
        return display


@dataclass
class Recommendation:
    """Product recommendation."""
    product: Product
    
    # Scoring
    match_score: float = 0.0  # 0-100
    
    # Reasoning
    match_reasons: list[str] = field(default_factory=list)
    
    # Deals
    applicable_deals: list[Deal] = field(default_factory=list)
    final_price: float = 0.0
    
    # Reviews summary
    review_summary: Optional[ReviewSummary] = None
    
    # Alternatives
    similar_products: list[str] = field(default_factory=list)
    
    # Ranking
    rank: int = 0
    
    def to_display(self) -> str:
        display = f"### {self.rank}. {self.product.name}\n"
        display += f"**{self.product.brand}** | "
        
        if self.final_price < self.product.price:
            display += f"~~${self.product.price:.2f}~~ → **${self.final_price:.2f}**\n"
        else:
            display += f"**${self.product.price:.2f}**\n"
        
        display += f"⭐ {self.product.rating}/5 ({self.product.review_count:,} reviews)\n"
        
        # Match reasons
        display += f"**Why we recommend:** {', '.join(self.match_reasons[:3])}\n"
        
        # Deals
        for deal in self.applicable_deals[:2]:
            display += f"🏷️ {deal.description}"
            if deal.code:
                display += f" (code: {deal.code})"
            display += "\n"
        
        # Sustainability
        if self.product.is_sustainable:
            certs = ", ".join(self.product.sustainability_certifications[:2])
            display += f"♻️ {certs}\n"
        
        return display


@dataclass
class ShoppingSession:
    """A shopping session with context."""
    session_id: str
    user_preferences: UserPreferences
    
    # Current search
    query: str = ""
    category: Optional[Category] = None
    budget: Optional[float] = None
    
    # Constraints
    must_have: list[str] = field(default_factory=list)
    nice_to_have: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    
    # Results
    products_found: list[Product] = field(default_factory=list)
    deals_found: list[Deal] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)
    
    # Conversation
    messages: list[dict] = field(default_factory=list)
    
    # Status
    created_at: str = ""
    updated_at: str = ""
    
    def add_message(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })


@dataclass
class ShoppingResult:
    """Final shopping results."""
    query: str
    
    # Top recommendations
    recommendations: list[Recommendation] = field(default_factory=list)
    
    # Available deals
    deals: list[Deal] = field(default_factory=list)
    
    # Price insights
    price_range: tuple = (0.0, 0.0)
    average_price: float = 0.0
    best_value: Optional[Recommendation] = None
    
    # Summary
    total_products_searched: int = 0
    retailers_searched: list[str] = field(default_factory=list)
    
    def to_display(self) -> str:
        display = f"# Shopping Results: {self.query}\n\n"
        display += f"Searched {self.total_products_searched} products across {len(self.retailers_searched)} retailers\n\n"
        
        display += "## Top Recommendations\n\n"
        for rec in self.recommendations[:5]:
            display += rec.to_display() + "\n"
        
        if self.deals:
            display += "\n## Available Deals\n\n"
            for deal in self.deals[:5]:
                display += deal.to_display() + "\n"
        
        return display
