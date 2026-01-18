"""
E-commerce Personal Shopper - CrewAI Shopping Crew 

Parallel agent architecture for comprehensive shopping assistance.

                    ┌─────────────────────┐
                    │   CONCIERGE         │
                    │   (Main Interface)  │
                    └──────────┬──────────┘
                               │
    ┌──────────────────────────┼──────────────────────────┐
    │                          │                          │
    ▼                          ▼                          ▼
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│   STYLE     │         │   SEARCH    │         │   DEALS     │
│   ADVISOR   │         │   AGENT     │         │   FINDER    │
└──────┬──────┘         └──────┬──────┘         └──────┬──────┘
       │                       │                       │
       │         ┌─────────────┴─────────────┐         │
       │         │                           │         │
       │         ▼                           ▼         │
       │   ┌─────────────┐           ┌─────────────┐   │
       │   │   PRICE     │           │   REVIEW    │   │
       │   │   COMPARE   │           │   ANALYZER  │   │
       │   └──────┬──────┘           └──────┬──────┘   │
       │          │                         │          │
       └──────────┴────────────┬────────────┴──────────┘
                               │
                               ▼
                        ┌─────────────┐
                        │ RECOMMENDER │
                        └─────────────┘
"""

from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

try:
    from crewai import Crew, Task, Process
except ImportError:
    raise ImportError("Install crewai: pip install crewai")

from .agents import ShoppingTeam
from .data import product_db
from .models import (
    UserPreferences, ShoppingSession, ShoppingResult, Recommendation,
    Product, Deal, Category
)


@dataclass
class ShopperConfig:
    """Configuration for the personal shopper."""
    llm_provider: str = "gemini"
    llm_model: Optional[str] = None
    verbose: bool = True
    parallel_execution: bool = True


@dataclass
class ShoppingRequest:
    """A shopping request from the user."""
    query: str
    budget: Optional[float] = None
    preferences: Optional[UserPreferences] = None
    
    # Specific requirements
    category: Optional[str] = None
    must_have: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    
    # Flags
    sustainable_only: bool = False
    deals_only: bool = False


class PersonalShopperCrew:
    """
    Personal Shopper Crew
    
    A parallel agent system for comprehensive shopping assistance:
    - Understanding user preferences
    - Searching across retailers
    - Finding deals and comparing prices
    - Analyzing reviews
    - Making personalized recommendations
    
    Usage:
        shopper = PersonalShopperCrew()
        result = shopper.shop("winter jacket, budget $200, sustainable")
        
        for rec in result.recommendations:
            print(rec.to_display())
    """
    
    def __init__(self, config: Optional[ShopperConfig] = None):
        self.config = config or ShopperConfig()
        
        # Initialize team
        self.team = ShoppingTeam(
            llm_provider=self.config.llm_provider,
            llm_model=self.config.llm_model,
        )
    
    def shop(self, query: str, budget: float = None, sustainable_only: bool = False) -> ShoppingResult:
        """
        Main shopping interface.
        
        Args:
            query: What the user is looking for
            budget: Maximum budget
            sustainable_only: Only show sustainable products
        
        Returns:
            ShoppingResult with recommendations and deals
        """
        import time
        start_time = time.time()
        
        # Parse the query for additional context
        query_lower = query.lower()
        
        # Detect sustainability preference from query
        if any(word in query_lower for word in ["sustainable", "eco", "green", "ethical"]):
            sustainable_only = True
        
        # Detect budget from query
        if budget is None:
            import re
            budget_match = re.search(r'\$(\d+)', query)
            if budget_match:
                budget = float(budget_match.group(1))
        
        # Create tasks for parallel execution
        tasks = self._create_shopping_tasks(query, budget, sustainable_only)
        
        # Create crew with parallel process
        crew = Crew(
            agents=[
                self.team.concierge,
                self.team.style_advisor,
                self.team.search_agent,
                self.team.deals_finder,
                self.team.review_analyzer,
                self.team.recommender,
            ],
            tasks=tasks,
            process=Process.sequential,  # Tasks are designed to flow sequentially
            verbose=self.config.verbose,
        )
        
        # Execute
        try:
            result = crew.kickoff()
            
            # Generate shopping result
            shopping_result = self._create_shopping_result(query, budget, sustainable_only, result)
            
            return shopping_result
            
        except Exception as e:
            print(f"Error during shopping: {e}")
            # Return basic result from direct search
            return self._fallback_search(query, budget, sustainable_only)
    
    def _create_shopping_tasks(self, query: str, budget: float, sustainable_only: bool) -> list[Task]:
        """Create the shopping task pipeline."""
        
        budget_str = f"${budget:.2f}" if budget else "flexible"
        sustainable_str = "YES - only sustainable products" if sustainable_only else "not required"
        
        # Task 1: Concierge understands the request
        task_understand = Task(
            description=f"""Understand the customer's shopping request and identify their needs.

Request: "{query}"
Budget: {budget_str}
Sustainable only: {sustainable_str}

1. Identify the product category they're looking for
2. Note any specific features or requirements mentioned
3. Understand their style preferences if mentioned
4. Summarize what they need in a clear brief for the team""",
            expected_output="Clear shopping brief with identified needs and preferences",
            agent=self.team.concierge,
        )
        
        # Task 2: Style advisor analyzes preferences
        task_style = Task(
            description=f"""Analyze the customer's style preferences based on their request.

Request: "{query}"
Sustainable preference: {sustainable_str}

1. Use the style matching tool to identify products that match their preferences
2. Consider sustainability if important to them
3. Identify style themes (minimalist, bold, classic, etc.)
4. Note any specific brands or aesthetics they might prefer""",
            expected_output="Style analysis with matching product suggestions",
            agent=self.team.style_advisor,
        )
        
        # Task 3: Search agent finds products
        task_search = Task(
            description=f"""Search for products matching the customer's request.

Request: "{query}"
Budget: {budget_str}
Sustainable only: {sustainable_str}

1. Search for products matching the query
2. Filter by budget if specified
3. Filter for sustainable products if required
4. Find at least 5-10 relevant options
5. Note product IDs for further analysis""",
            expected_output="List of relevant products with IDs and basic info",
            agent=self.team.search_agent,
        )
        
        # Task 4: Deals finder looks for savings
        task_deals = Task(
            description=f"""Find deals and discounts for the products found.

1. Search for current deals and coupons
2. Check which deals apply to the found products
3. Calculate potential savings
4. Note the best discount codes available""",
            expected_output="List of applicable deals with potential savings",
            agent=self.team.deals_finder,
        )
        
        # Task 5: Review analyzer checks quality
        task_reviews = Task(
            description=f"""Analyze reviews for the top product candidates.

1. Get review summaries for the top 3-5 products
2. Identify pros and cons from customer feedback
3. Check fit information if applicable
4. Note any quality concerns""",
            expected_output="Review analysis with pros, cons, and fit info",
            agent=self.team.review_analyzer,
        )
        
        # Task 6: Recommender creates final recommendations
        task_recommend = Task(
            description=f"""Create final personalized recommendations.

Based on all the research:
1. Select the top 3 products that best match the customer's needs
2. Generate detailed recommendations for each
3. Include applicable deals and final prices
4. Explain why each product is recommended
5. Rank them in order of best fit

Format as clear, actionable recommendations.""",
            expected_output="Top 3 personalized product recommendations with reasoning",
            agent=self.team.recommender,
        )
        
        return [task_understand, task_style, task_search, task_deals, task_reviews, task_recommend]
    
    def _create_shopping_result(self, query: str, budget: float, sustainable_only: bool, crew_result) -> ShoppingResult:
        """Create ShoppingResult from crew execution."""
        
        # Get products from database (this would come from the crew in production)
        products = product_db.search_products(
            query=query.split(",")[0].strip(),  # Use first part of query
            max_price=budget or 1000,
            sustainable_only=sustainable_only,
        )
        
        # Get deals
        all_deals = [d for d in product_db.deals if d.is_active]
        
        # Create recommendations
        recommendations = []
        for i, product in enumerate(products[:5], 1):
            # Get applicable deals
            product_deals = product_db.get_deals_for_product(product)
            
            # Calculate final price
            final_price = product.price
            for deal in product_deals:
                discounted = deal.calculate_discount(product.price)
                if discounted < final_price:
                    final_price = discounted
            
            # Get reviews
            review_summary = product_db.get_reviews(product.product_id)
            
            # Create match reasons
            reasons = []
            if product.rating >= 4.5:
                reasons.append("Highly rated")
            if product.is_sustainable:
                reasons.append("Sustainable")
            if product_deals:
                reasons.append("Has deals available")
            if product.ships_free:
                reasons.append("Free shipping")
            if not reasons:
                reasons.append("Matches your search")
            
            recommendations.append(Recommendation(
                product=product,
                match_score=90 - (i * 5),  # Decreasing score by rank
                match_reasons=reasons,
                applicable_deals=product_deals,
                final_price=final_price,
                review_summary=review_summary,
                rank=i,
            ))
        
        # Find best value
        best_value = min(recommendations, key=lambda r: r.final_price) if recommendations else None
        
        # Get price range
        prices = [r.final_price for r in recommendations] if recommendations else [0]
        
        return ShoppingResult(
            query=query,
            recommendations=recommendations,
            deals=all_deals[:5],
            price_range=(min(prices), max(prices)),
            average_price=sum(prices) / len(prices) if prices else 0,
            best_value=best_value,
            total_products_searched=len(product_db.products),
            retailers_searched=[r.value for r in set(p.retailer for p in products)],
        )
    
    def _fallback_search(self, query: str, budget: float, sustainable_only: bool) -> ShoppingResult:
        """Fallback to direct search if crew fails."""
        return self._create_shopping_result(query, budget, sustainable_only, None)
    
    def quick_search(self, query: str, max_results: int = 5) -> list[Product]:
        """Quick search without full crew execution."""
        return product_db.search_products(query=query)[:max_results]
    
    def find_deals(self, category: str = None) -> list[Deal]:
        """Find current deals."""
        deals = [d for d in product_db.deals if d.is_active]
        
        if category:
            try:
                cat = Category(category.lower())
                deals = [d for d in deals if cat in d.applicable_categories or not d.applicable_categories]
            except ValueError:
                pass
        
        return deals
    
    def get_product_recommendation(self, product_id: str) -> Optional[Recommendation]:
        """Get a detailed recommendation for a specific product."""
        product = product_db.get_product(product_id)
        
        if not product:
            return None
        
        deals = product_db.get_deals_for_product(product)
        reviews = product_db.get_reviews(product_id)
        
        final_price = product.price
        for deal in deals:
            discounted = deal.calculate_discount(product.price)
            if discounted < final_price:
                final_price = discounted
        
        return Recommendation(
            product=product,
            match_score=85,
            match_reasons=["Detailed recommendation"],
            applicable_deals=deals,
            final_price=final_price,
            review_summary=reviews,
            rank=1,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def shop(query: str, budget: float = None, sustainable_only: bool = False) -> ShoppingResult:
    """
    Quick shopping function.
    
    Args:
        query: What you're looking for
        budget: Maximum budget
        sustainable_only: Only sustainable products
    
    Returns:
        ShoppingResult with recommendations
    """
    shopper = PersonalShopperCrew()
    return shopper.shop(query, budget, sustainable_only)


def quick_search(query: str) -> list[Product]:
    """Quick product search."""
    return product_db.search_products(query=query)[:5]


def find_deals() -> list[Deal]:
    """Find current deals."""
    return [d for d in product_db.deals if d.is_active]
