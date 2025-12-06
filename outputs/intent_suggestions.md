# Intent Expansion Pipeline Report

## Summary

- **Total messages analyzed**: 200
- **Number of clusters**: 15
- **Clustering quality (silhouette)**: 0.301
- **LLM Refinement**: Enabled ✓
- **Existing intents in mapper**: 35

### Recommendation Breakdown

- 🆕 **NEW_INTENT_CANDIDATE**: 1
- ✂️ **CONSIDER_SPLIT**: 1
- ✅ **MAPS_TO_EXISTING**: 2

---

## Detailed Suggestions

### 1. 🆕 NEW_INTENT_CANDIDATE: `product_specific_usage`

**Description**: This intent captures customer inquiries about the usage, application, or special features of a specific product, such as the Rosemary & Rice Water Hair Growth Spray.

- **Cluster size**: 9 (4.5% of messages)
- **Top terms**: spray, rice, rice water, water, water hair, growth spray

**Justification**: Cluster does not match any existing intent well (best similarity=0.09). Potential new intent covering 9 messages.

**Example messages**:
  - "Add to cart - Rosemary & Rice Water Hair Growth Spray"
  - "What's special about it? Rosemary & Rice Water Hair Growth Spray"
  - "How to use it in daily routine? Rosemary & Rice Water Hair Growth Spray"

### 2. ✂️ CONSIDER_SPLIT: `product_specialty_request`

**Description**: This intent captures customer inquiries about the specific benefits, features, or suitability of a product for their needs, often related to hair or skin concerns.

- **Cluster size**: 120 (60.0% of messages)
- **Top terms**: shampoo, oil, body, hai, hair, special
- **Best match**: Recommendation → Seeking Solution (similarity: 0.207)

**Justification**: Cluster partially overlaps with 'Seeking Solution' (similarity=0.21). Consider creating a sub-intent.

**Example messages**:
  - "Best for oily skin"
  - "Mera order confirm ho gaya hai"
  - "Others details "

### 3. ✅ MAPS_TO_EXISTING: `order_cancel_rto`

- **Cluster size**: 14 (7.0% of messages)
- **Top terms**: order, cancel, rto, 10, means, refund
- **Best match**: About Product → Place Order (similarity: 0.978)

**Justification**: Cluster maps well to existing intent 'Place Order' (similarity=0.98).

**Example messages**:
  - "When the order will be shipped"
  - "My order company name "
  - "where is my order"

### 4. ✅ MAPS_TO_EXISTING: `track_order_track_order`

- **Cluster size**: 11 (5.5% of messages)
- **Top terms**: track order, track, order, water hair, water 30, water
- **Best match**: About Product → Place Order (similarity: 0.492)

**Justification**: Cluster maps well to existing intent 'Place Order' (similarity=0.49).

**Example messages**:
  - "How to track order?"
  - "How to track order?"
  - "How to track order"

---

## Existing Intent Mapper Reference

### Basic Interactions

- **Greetings** (`greetings`): When a customer gives greetings only like Hi, hello, thank you etc.
- **Acknowledgment** (`acknowledgment`): When a customer gives acknowledgment only like Ok, Thanks, Got it, Alright etc.

### About Company

- **Contact Details** (`contact_details`): When a customer asks about the company's contact specifically.
- **About Team** (`about_team`): When a customer asks about the team.

### About Product

- **Unsure About Effectiveness** (`effectiveness`): When the customer is not sure about the effectiveness of the product.
- **Product Ingredients** (`product_ingredients`): When customer asks about the ingredients of a product.
- **Pricing** (`pricing`): When the message is related to the price of a product.
- **No More Product Query** (`no_query_uc`): When the customer's message indicates they have no further questions about the p...
- **All Products** (`all_product`): When the message is related to all products that company provides.
- **Payment Options Available** (`payment_options_available`): When the customer asks about payment options available.
- **Certification** (`certification`): When customer asks for product certification or if it is harmful chemical free.
- **Place Order** (`place_order`): When the customer asks how to place an order.
- **Comparison** (`comparison`): When the message is related to comparing one product with another product.
- **Out of Stock** (`out_of_stock`): When customer says product is out of stock or can't find it on website.
- **Product Info** (`product_info`): When the customer wants to know how to use etc or says yes to a question about h...
- **Product Specialty** (`product_speciality_uc`): When customer asks specialty or benefits of a product or says yes to a question ...

### Recommendation

- **Customer Problem** (`customer_problem`): When the customer talks about a personal issue, concern, or challenge they are f...
- **Customer Concern Related Information** (`customer_information`): When the customer shares their details like concern etc for product recommendati...
- **Seeking Solution** (`seeking_solution`): When customer asks recommendation for a product/category for their skin/hair typ...

### Logistics

- **Address Change** (`address_change`): When the customer asks about changing the address of the order.
- **Wrong/Damaged Order** (`wrong_order`): When the customer mentions that they received a wrong order/incorrect order or d...
- **Marked Delivered, Not Received** (`order_delivered_but_not_received`): Only applicable when the customer reports that the tracking/portal shows their o...
- **B2B Queries** (`b2b_queries`): When the customer asks about B2B queries like placing bulk order or talks about ...
- **Refund Policy** (`refund_policy`): When a customer asks for the status of their refund/or when they will get their ...
- **Refund Required** (`refund_required`): When a customer asking you to give refund or asking about refund policy.
- **Cancellation** (`cancellation`): When the customer asks about the cancellation policy for any product.
- **Payment Related Issues** (`payment`): When the customer talks about payment is failed but their money is deducted, or ...
- **E-commerce/Quick Commerce** (`ecommerce_quick_commerce`): When the customer asks about the availability of the product on e-commerce platf...
- **Feedback** (`feedback`): When the customer is complaining or giving feedback about a product/service or d...
- **Order Status** (`order_status`): When the customer talks about the status of an order, or asks for order tracking...
- **Discount** (`discount`): When the customer asks about offers or discounts available for a product.
- **Talk to Agent** (`talk_to_agent`): When the customer is explicitly asking to wants to connect with a live agent or ...
- **Return Policy** (`return_policy`): When a customer asks for the return policy for any product.
- **Complaint** (`complaint`): When the customer expresses a complaint, concern, dissatisfaction, frustration, ...
- **Website Issue** (`website_issue`): When the customer reports any issue related to the website, such as errors, load...
