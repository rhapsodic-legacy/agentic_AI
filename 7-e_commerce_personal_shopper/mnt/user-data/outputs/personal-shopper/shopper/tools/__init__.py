"""
E-commerce Personal Shopper - CrewAI Tools

Tools for:
- Product search
- Price comparison
- Deal finding
- Review analysis
- Style matching
"""

from typing import Optional, Type
from pydantic import BaseModel, Field

try:
    from crewai.tools import BaseTool
except ImportError:
    class BaseTool:
        name: str = ""
        description: str = ""
        def _run(self, *args, **kwargs):
            raise NotImplementedError

from ..data import product_db
from ..models import (
    Product, Deal, Category, Retailer, UserPreferences,
    Recommendation, ReviewSummary, PriceComparison
)


# =============================================================================
# Input Schemas
# =============================================================================

class SearchInput(BaseModel):
    """Input for product search."""
    query: str = Field(..., description="Search query (e.g., 'winter jacket')")
    max_price: float = Field(default=1000, description="Maximum price")
    sustainable_only: bool = Field(default=False, description="Only sustainable products")


class ProductInput(BaseModel):
    """Input for product-specific operations."""
    product_id: str = Field(..., description="Product ID")


class PriceCompareInput(BaseModel):
    """Input for price comparison."""
    product_name: str = Field(..., description="Product name to compare")
    brand: str = Field(..., description="Brand name")


class StyleInput(BaseModel):
    """Input for style matching."""
    preferences: str = Field(..., description="User preferences description")
    category: str = Field(default="", description="Product category")


# =============================================================================
# Search Tools
# =============================================================================

class SearchProductsTool(BaseTool):
    name: str = "search_products"
    description: str = """Search for products across multiple retailers.
    Input: search query, optional max_price, optional sustainable_only flag.
    Returns list of matching products with prices, ratings, and availability."""
    args_schema: Type[BaseModel] = SearchInput
    
    def _run(self, query: str, max_price: float = 1000, sustainable_only: bool = False) -> str:
        products = product_db.search_products(
            query=query,
            max_price=max_price,
            sustainable_only=sustainable_only,
        )
        
        if not products:
            return f"No products found matching '{query}'"
        
        result = f"Found {len(products)} products matching '{query}':\n\n"
        
        for i, product in enumerate(products[:10], 1):
            result += f"{i}. **{product.name}** by {product.brand}\n"
            result += f"   💰 ${product.price:.2f}"
            if product.original_price and product.original_price > product.price:
                result += f" (was ${product.original_price:.2f})"
            result += f"\n   ⭐ {product.rating}/5 ({product.review_count:,} reviews)\n"
            result += f"   🏪 {product.retailer.value.title()}"
            if product.is_sustainable:
                result += " | ♻️ Sustainable"
            result += f"\n   ID: {product.product_id}\n\n"
        
        return result


class SearchByCategory(BaseTool):
    name: str = "search_by_category"
    description: str = """Search products by category.
    Categories: clothing, electronics, home, beauty, sports, outdoor, accessories, footwear"""
    
    def _run(self, category: str, max_price: float = 1000) -> str:
        try:
            cat = Category(category.lower())
        except ValueError:
            return f"Invalid category. Choose from: {[c.value for c in Category]}"
        
        products = product_db.search_products(category=cat, max_price=max_price)
        
        if not products:
            return f"No products found in {category} category"
        
        result = f"Products in {category}:\n\n"
        for product in products[:10]:
            result += f"• {product.name} by {product.brand} - ${product.price:.2f}\n"
        
        return result


class GetProductDetailsTool(BaseTool):
    name: str = "get_product_details"
    description: str = """Get detailed information about a specific product.
    Input: product_id. Returns full product details including features, materials, and availability."""
    args_schema: Type[BaseModel] = ProductInput
    
    def _run(self, product_id: str) -> str:
        product = product_db.get_product(product_id)
        
        if not product:
            return f"Product not found: {product_id}"
        
        result = f"""**{product.name}** by {product.brand}

**Price:** ${product.price:.2f}"""
        
        if product.original_price and product.original_price > product.price:
            savings = product.original_price - product.price
            result += f" (Save ${savings:.2f} - was ${product.original_price:.2f})"
        
        result += f"""

**Rating:** ⭐ {product.rating}/5 based on {product.review_count:,} reviews

**Description:** {product.description}

**Features:**
"""
        for feature in product.features:
            result += f"• {feature}\n"
        
        result += f"""
**Materials:** {', '.join(product.materials) if product.materials else 'N/A'}

**Available Sizes:** {', '.join(product.available_sizes) if product.available_sizes else 'N/A'}
**Available Colors:** {', '.join(product.available_colors) if product.available_colors else 'N/A'}

**Retailer:** {product.retailer.value.title()}
**In Stock:** {'Yes ✓' if product.in_stock else 'No ✗'}
**Free Shipping:** {'Yes ✓' if product.ships_free else 'No'}
"""
        
        if product.is_sustainable:
            result += f"\n♻️ **Sustainability:** {', '.join(product.sustainability_certifications)}"
        
        return result


# =============================================================================
# Deal & Price Tools
# =============================================================================

class FindDealsTool(BaseTool):
    name: str = "find_deals"
    description: str = """Find current deals, coupons, and discounts.
    Can filter by retailer or category."""
    
    def _run(self, retailer: str = "", category: str = "") -> str:
        deals = product_db.deals
        
        # Filter
        if retailer:
            try:
                ret = Retailer(retailer.lower())
                deals = [d for d in deals if d.retailer == ret or d.retailer is None]
            except ValueError:
                pass
        
        if category:
            try:
                cat = Category(category.lower())
                deals = [d for d in deals if cat in d.applicable_categories or not d.applicable_categories]
            except ValueError:
                pass
        
        active_deals = [d for d in deals if d.is_active]
        
        if not active_deals:
            return "No active deals found"
        
        result = "**Current Deals & Coupons:**\n\n"
        
        for deal in active_deals:
            result += f"🏷️ **{deal.description}**\n"
            if deal.code:
                result += f"   Code: `{deal.code}`\n"
            if deal.discount_percent:
                result += f"   Save: {deal.discount_percent}% off\n"
            elif deal.discount_amount:
                result += f"   Save: ${deal.discount_amount:.2f}\n"
            if deal.minimum_purchase:
                result += f"   Min purchase: ${deal.minimum_purchase:.2f}\n"
            if deal.valid_until:
                result += f"   Expires: {deal.valid_until}\n"
            result += "\n"
        
        return result


class GetDealsForProductTool(BaseTool):
    name: str = "get_deals_for_product"
    description: str = """Find applicable deals for a specific product.
    Input: product_id. Returns all coupons and discounts that apply."""
    args_schema: Type[BaseModel] = ProductInput
    
    def _run(self, product_id: str) -> str:
        product = product_db.get_product(product_id)
        
        if not product:
            return f"Product not found: {product_id}"
        
        deals = product_db.get_deals_for_product(product)
        
        if not deals:
            return f"No special deals found for {product.name}"
        
        result = f"**Deals for {product.name}:**\n\n"
        result += f"Original price: ${product.price:.2f}\n\n"
        
        best_price = product.price
        
        for deal in deals:
            discounted = deal.calculate_discount(product.price)
            savings = product.price - discounted
            
            if discounted < best_price:
                best_price = discounted
            
            result += f"🏷️ {deal.description}\n"
            if deal.code:
                result += f"   Code: `{deal.code}` → ${discounted:.2f} (save ${savings:.2f})\n"
            else:
                result += f"   Price: ${discounted:.2f} (save ${savings:.2f})\n"
            result += "\n"
        
        result += f"**Best possible price: ${best_price:.2f}**"
        
        return result


class ComparePricesTool(BaseTool):
    name: str = "compare_prices"
    description: str = """Compare prices for a product across different retailers.
    Input: product_name and brand."""
    args_schema: Type[BaseModel] = PriceCompareInput
    
    def _run(self, product_name: str, brand: str) -> str:
        comparison = product_db.get_price_comparison(product_name, brand)
        
        if not comparison.prices:
            return f"No price data found for {brand} {product_name}"
        
        return comparison.to_display()


# =============================================================================
# Review Tools
# =============================================================================

class GetReviewsTool(BaseTool):
    name: str = "get_reviews"
    description: str = """Get review summary for a product including pros, cons, and fit info.
    Input: product_id."""
    args_schema: Type[BaseModel] = ProductInput
    
    def _run(self, product_id: str) -> str:
        reviews = product_db.get_reviews(product_id)
        
        if not reviews:
            product = product_db.get_product(product_id)
            if product:
                return f"No detailed reviews available for {product.name}. Rating: {product.rating}/5 ({product.review_count} reviews)"
            return f"Product not found: {product_id}"
        
        return reviews.to_display()


class AnalyzeReviewSentimentTool(BaseTool):
    name: str = "analyze_review_sentiment"
    description: str = """Analyze the sentiment and key themes from product reviews.
    Input: product_id."""
    args_schema: Type[BaseModel] = ProductInput
    
    def _run(self, product_id: str) -> str:
        reviews = product_db.get_reviews(product_id)
        product = product_db.get_product(product_id)
        
        if not product:
            return f"Product not found: {product_id}"
        
        result = f"**Review Analysis for {product.name}:**\n\n"
        
        if reviews:
            result += f"Overall Sentiment: {reviews.overall_sentiment.upper()}\n\n"
            
            result += "**What customers love:**\n"
            for pro in reviews.pros:
                result += f"✅ {pro}\n"
            
            result += "\n**Potential concerns:**\n"
            for con in reviews.cons:
                result += f"⚠️ {con}\n"
            
            if reviews.fit_feedback:
                result += f"\n**Fit:** {reviews.fit_feedback}"
            
            if reviews.most_helpful_positive:
                result += f"\n\n**Top review:** \"{reviews.most_helpful_positive}\""
        else:
            result += f"Rating: {product.rating}/5 based on {product.review_count} reviews"
        
        return result


# =============================================================================
# Style & Recommendation Tools
# =============================================================================

class StyleMatchTool(BaseTool):
    name: str = "match_style"
    description: str = """Match products to user style preferences.
    Input: preferences description (e.g., "minimalist, sustainable, neutral colors")"""
    args_schema: Type[BaseModel] = StyleInput
    
    def _run(self, preferences: str, category: str = "") -> str:
        pref_lower = preferences.lower()
        
        # Parse preferences
        sustainable = any(word in pref_lower for word in ["sustainable", "eco", "green", "ethical"])
        
        all_products = product_db.products
        
        if category:
            try:
                cat = Category(category.lower())
                all_products = [p for p in all_products if p.category == cat]
            except ValueError:
                pass
        
        # Score products
        scored = []
        for product in all_products:
            score = 0
            reasons = []
            
            # Sustainability match
            if sustainable and product.is_sustainable:
                score += 30
                reasons.append("Sustainable product")
            
            # Tag matching
            product_text = f"{product.name} {product.brand} {' '.join(product.tags)} {' '.join(product.materials)}".lower()
            
            preference_words = pref_lower.split()
            for word in preference_words:
                if word in product_text:
                    score += 10
                    reasons.append(f"Matches '{word}'")
            
            # Rating bonus
            if product.rating >= 4.5:
                score += 15
                reasons.append("Highly rated")
            
            if score > 0:
                scored.append((product, score, reasons[:3]))
        
        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        if not scored:
            return f"No products found matching preferences: {preferences}"
        
        result = f"**Products matching your style:**\n\n"
        
        for product, score, reasons in scored[:5]:
            result += f"**{product.name}** by {product.brand}\n"
            result += f"   ${product.price:.2f} | ⭐ {product.rating}/5\n"
            result += f"   Why: {', '.join(reasons)}\n"
            result += f"   ID: {product.product_id}\n\n"
        
        return result


class GenerateRecommendationTool(BaseTool):
    name: str = "generate_recommendation"
    description: str = """Generate a personalized product recommendation with deals.
    Input: product_id to analyze and recommend."""
    args_schema: Type[BaseModel] = ProductInput
    
    def _run(self, product_id: str) -> str:
        product = product_db.get_product(product_id)
        
        if not product:
            return f"Product not found: {product_id}"
        
        # Get deals
        deals = product_db.get_deals_for_product(product)
        
        # Calculate final price
        final_price = product.price
        for deal in deals:
            discounted = deal.calculate_discount(product.price)
            if discounted < final_price:
                final_price = discounted
        
        # Get reviews
        reviews = product_db.get_reviews(product_id)
        
        # Build recommendation
        result = f"""## Recommendation: {product.name}

**{product.brand}** | ${final_price:.2f}"""
        
        if final_price < product.price:
            result += f" (was ${product.price:.2f})"
        
        result += f"""

⭐ {product.rating}/5 ({product.review_count:,} reviews)
"""
        
        # Reviews summary
        if reviews:
            result += f"\n**What people say:** \"{reviews.most_helpful_positive}\"\n"
            if reviews.fit_feedback:
                result += f"**Fit:** {reviews.fit_feedback}\n"
        
        # Deals
        if deals:
            result += "\n**Available deals:**\n"
            for deal in deals[:2]:
                result += f"🏷️ {deal.description}"
                if deal.code:
                    result += f" (code: `{deal.code}`)"
                result += "\n"
        
        # Sustainability
        if product.is_sustainable:
            result += f"\n♻️ **Sustainability:** {', '.join(product.sustainability_certifications)}\n"
        
        # Features
        result += "\n**Key features:**\n"
        for feature in product.features[:4]:
            result += f"• {feature}\n"
        
        result += f"\n**Available at:** {product.retailer.value.title()}"
        if product.ships_free:
            result += " | 🚚 Free shipping"
        
        return result


# =============================================================================
# Tool Factory
# =============================================================================

def get_all_tools() -> list:
    """Get all available tools."""
    return [
        SearchProductsTool(),
        SearchByCategory(),
        GetProductDetailsTool(),
        FindDealsTool(),
        GetDealsForProductTool(),
        ComparePricesTool(),
        GetReviewsTool(),
        AnalyzeReviewSentimentTool(),
        StyleMatchTool(),
        GenerateRecommendationTool(),
    ]


def get_tools_for_role(role: str) -> list:
    """Get tools for a specific agent role."""
    role_tools = {
        "concierge": [SearchProductsTool(), GetProductDetailsTool(), GenerateRecommendationTool()],
        "style_advisor": [StyleMatchTool(), SearchProductsTool(), GetProductDetailsTool()],
        "search_agent": [SearchProductsTool(), SearchByCategory(), GetProductDetailsTool()],
        "deals_finder": [FindDealsTool(), GetDealsForProductTool(), ComparePricesTool()],
        "price_compare": [ComparePricesTool(), GetDealsForProductTool()],
        "review_analyzer": [GetReviewsTool(), AnalyzeReviewSentimentTool()],
        "recommender": [GenerateRecommendationTool(), GetProductDetailsTool(), GetReviewsTool(), GetDealsForProductTool()],
    }
    
    return role_tools.get(role, get_all_tools())
